"""Deterministic human audit and private offline worksheet. No automated judgments."""
from collections import Counter, defaultdict
from datetime import date
import json
from pathlib import Path
import re

from .constants import LABELS
from .data_policy import POLICY_SHA256, row_order
from .provenance import json_bytes, sha256_bytes

LANGUAGES = ("english", "mixed_uncertain", "non_english")
USABILITY = ("usable", "unusable")
ANNOTATIONS = ("plausible", "questionable", "uncertain")


def audit_sample(rows):
    strata = defaultdict(list)
    for row in rows:
        strata[row.event_id, row.y_true].append(row)
    selected = set()
    for (event, label), values in sorted(strata.items()):
        def rank(row):
            value = f"audit-v1|42|{event}|{LABELS[label]}|{row.numeric_id}|{row.source_key}"
            return sha256_bytes(value.encode("utf-8")), row.source_key
        selected.update(r.source_key for r in sorted(values, key=rank)[:3])
    rare = {r.source_key for r in rows if LABELS[r.y_true] == "missing_or_found_people"}
    sample = []
    for row in sorted(rows, key=row_order):
        if row.source_key in selected | rare:
            sample.append({**row.metadata(), "text": row.raw_text,
                           "class_label": LABELS[row.y_true],
                           "sample_reasons": [reason for reason, keys in
                                              (("stratified", selected), ("rare_census", rare))
                                              if row.source_key in keys]})
    return sample, {"sample_rows": len(sample), "stratified_rows": len(selected),
                    "rare_census_rows": len(rare), "overlap_rows": len(selected & rare)}


def audit_binding(sample, archive_sha256, kind):
    content = b"".join(json_bytes(row) for row in sample)
    binding = {"schema_version": 1, "policy_sha256": POLICY_SHA256,
               "archive_sha256": archive_sha256, "sample_sha256": sha256_bytes(content),
               "corpus_kind": kind}
    return content, {**binding, "audit_id": sha256_bytes(json_bytes(binding))}


def decision_template(sample, binding):
    return {**binding, "reviewer": "", "reviewed_at": "", "reviewer_count": 1,
            "protocol_change_requested": None, "protocol_change_reason": "",
            "rows": [{"source_key": row["source_key"], "language": None,
                      "usability": None, "annotation": None, "reason_codes": [], "notes": ""}
                     for row in sample]}


def validate_decisions(value, sample, binding):
    template = decision_template(sample, binding)
    if not isinstance(value, dict) or set(value) != set(template):
        raise ValueError("invalid audit decision structure")
    if any(value[k] != v for k, v in binding.items()):
        raise ValueError("stale audit decision binding")
    if type(value["reviewer_count"]) is not int or value["reviewer_count"] != 1:
        raise ValueError("audit requires the selected single-reviewer record")
    if not isinstance(value["reviewer"], str) or not value["reviewer"].strip():
        raise ValueError("audit reviewer is required")
    if not isinstance(value["reviewed_at"], str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value["reviewed_at"]):
        raise ValueError("audit date must be ISO calendar date")
    date.fromisoformat(value["reviewed_at"])
    if type(value["protocol_change_requested"]) is not bool or not isinstance(value["protocol_change_reason"], str):
        raise ValueError("explicit protocol disposition is required")
    if value["protocol_change_requested"]:
        raise ValueError("audit requests a protocol amendment; freeze stopped")
    if value["protocol_change_reason"].strip():
        raise ValueError("unresolved protocol-change reason")
    if not isinstance(value["rows"], list):
        raise ValueError("audit decisions must be a list")
    decisions = {}
    fields = set(template["rows"][0]) if template["rows"] else {
        "source_key", "language", "usability", "annotation", "reason_codes", "notes"}
    for record in value["rows"]:
        if not isinstance(record, dict) or set(record) != fields:
            raise ValueError("invalid audit row structure")
        key = record["source_key"]
        if not isinstance(key, str) or key in decisions:
            raise ValueError("duplicate or invalid audit decision key")
        if (record["language"] not in LANGUAGES or record["usability"] not in USABILITY
                or record["annotation"] not in ANNOTATIONS):
            raise ValueError("audit decisions are incomplete")
        codes = record["reason_codes"]
        if not isinstance(codes, list) or any(not isinstance(c, str) or not c.strip() for c in codes):
            raise ValueError("invalid audit reason codes")
        flagged = (record["language"] != "english" or record["usability"] != "usable"
                   or record["annotation"] != "plausible")
        if not isinstance(record["notes"], str) or (flagged and (not codes or not record["notes"].strip())):
            raise ValueError("flagged audit decisions require reasons and notes")
        decisions[key] = record
    if set(decisions) != {r["source_key"] for r in sample}:
        raise ValueError("audit decisions do not match the complete sample")
    return decisions


def apply_decisions(rows, decisions):
    retained, excluded = [], []
    for row in rows:
        decision = decisions.get(row.source_key)
        reasons = []
        if decision:
            if decision["language"] == "non_english":
                reasons.append("audit_non_english")
            if decision["usability"] == "unusable":
                reasons.append("audit_unusable")
        if reasons:
            excluded.append({**row.metadata(), "stage": "audit", "reasons": reasons})
        else:
            retained.append(row)
    return retained, excluded


def audit_summary(sample, decisions):
    result = []
    for event, label in sorted({(r["event_id"], r["y_true"]) for r in sample}):
        values = [decisions[r["source_key"]] for r in sample
                  if r["event_id"] == event and r["y_true"] == label]
        counts = Counter(f"{field}:{v[field]}" for v in values for field in ("language", "usability", "annotation"))
        result.append({"event_id": event, "class_label": LABELS[label],
                       "reviewed_rows": len(values), "findings": dict(sorted(counts.items()))})
    return {"reviewer_count": 1, "second_reviewer": False, "sample_rows": len(sample),
            "interpretation": "stratified sample plus rare-class census; not a corpus accuracy estimate",
            "event_class_findings": result}


def worksheet_html(sample, template):
    # Escape HTML-significant characters before embedding private JSON in a script.
    payload = json.dumps({"sample": sample, "template": template}, ensure_ascii=True).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return (_HTML.replace("__PAYLOAD__", payload)).encode("utf-8")


_HTML = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'none'; base-uri 'none'">
<title>Private HumAID audit</title>
<style>body{font:16px system-ui;max-width:920px;margin:32px auto;padding:0 20px;color:#172635;background:#f4f6f8}fieldset,article{background:white;border:1px solid #ccd5df;padding:18px;margin:16px 0;border-radius:8px}button,input,select,textarea{font:inherit;margin:6px;padding:8px}textarea{width:90%;height:65px}#text{white-space:pre-wrap;overflow-wrap:anywhere;font-size:19px;line-height:1.5}label{display:inline-block}#status{font-weight:600;color:#194e7d}.warning{color:#893d13}button{cursor:pointer}small{display:block}</style>
<h1>Private language and annotation audit</h1>
<p>Offline, single-reviewer worksheet. No network requests or automatic judgments. Save progress before closing; browser storage is not used. Do not upload or share this page.</p>
<p>First judge the language. The supplied label appears after a language choice. Keep mixed or uncertain language; exclude only confirmed non-English or unusable rows. Flag suspected annotation errors and retain supplied labels. Findings requiring broader changes stop the freeze.</p>
<div id="status"></div><article><small id="context"></small><p id="text"></p></article>
<fieldset><legend>1. Language</legend><select id="language"><option value="">Choose…</option><option value="english">English</option><option value="mixed_uncertain">Mixed / uncertain</option><option value="non_english">Confirmed non-English</option></select></fieldset>
<fieldset id="annotationPanel" hidden><legend>2. Usability and annotation</legend><p>Supplied label: <strong id="label"></strong></p>
<select id="usability"><option value="">Usability…</option><option value="usable">Usable</option><option value="unusable">Confirmed unusable</option></select>
<select id="annotation"><option value="">Annotation…</option><option value="plausible">Plausible</option><option value="questionable">Questionable</option><option value="uncertain">Uncertain</option></select>
<p>Review broad category fit, not factual truth or verified urgency. Notes are required for every flag.</p><textarea id="notes" placeholder="Private explanation for flags"></textarea></fieldset>
<button id="previous">Previous</button><button id="next">Save row / next</button><button id="pending">Next incomplete</button>
<fieldset><legend>Review record</legend><label>Reviewer <input id="reviewer"></label><label>Date <input type="date" id="date"></label><p><label><input type="checkbox" id="change">Findings require a broader protocol change (blocks freeze)</label></p><textarea id="changeReason" placeholder="Describe any broader exclusions or relabeling request"></textarea><p><label><input type="checkbox" id="attest">I personally reviewed all rows and the broader-change disposition.</label></p></fieldset>
<button id="save">Download progress JSON</button><button id="export">Download completed decisions</button><label>Restore progress <input type="file" id="restore" accept=".json"></label>
<p id="message" class="warning" role="status"></p>
<script>
const payload=__PAYLOAD__;
const sample=payload.sample;let decisions=structuredClone(payload.template),index=0;
const $=id=>document.getElementById(id);
const complete=r=>Boolean(r.language&&r.usability&&r.annotation)&&((r.language==='english'&&r.usability==='usable'&&r.annotation==='plausible')||r.notes.trim());
function draw(){const s=sample[index],r=decisions.rows[index];$('status').textContent=`Row ${index+1} of ${sample.length} · ${decisions.rows.filter(complete).length} complete`;$('context').textContent=s.event_id+' · '+s.sample_reasons.join(', ');$('text').textContent=s.text;for(const f of ['language','usability','annotation','notes'])$(f).value=r[f]||'';$('annotationPanel').hidden=!r.language;$('label').textContent=s.class_label;$('previous').disabled=index===0;}
function store(){const r=decisions.rows[index];for(const f of ['language','usability','annotation','notes'])r[f]=$(f).value||((f==='notes')?'':null);r.reason_codes=[];if(r.language&&r.language!=='english')r.reason_codes.push('language_'+r.language);if(r.usability==='unusable')r.reason_codes.push('confirmed_unusable');if(r.annotation&&r.annotation!=='plausible')r.reason_codes.push('annotation_'+r.annotation);decisions.reviewer=$('reviewer').value.trim();decisions.reviewed_at=$('date').value;decisions.protocol_change_requested=$('change').checked;decisions.protocol_change_reason=$('changeReason').value.trim();}
function download(name){const blob=new Blob([JSON.stringify(decisions,null,2)+'\n'],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('language').onchange=()=>{$('annotationPanel').hidden=!$('language').value;};
$('previous').onclick=()=>{store();index--;draw();};
$('next').onclick=()=>{store();if(!complete(decisions.rows[index])){$('message').textContent='Complete all choices and explain any flags before continuing.';return;}$('message').textContent='';if(index<sample.length-1)index++;draw();};
$('pending').onclick=()=>{store();const n=decisions.rows.findIndex(r=>!complete(r));if(n>=0)index=n;draw();};
$('save').onclick=()=>{store();download('audit-progress.json');};
$('export').onclick=()=>{store();if(decisions.rows.some(r=>!complete(r))||!decisions.reviewer||!decisions.reviewed_at||!$('attest').checked){$('message').textContent='Complete every row, reviewer/date, and personal-review confirmation.';return;}if(decisions.protocol_change_requested||decisions.protocol_change_reason){$('message').textContent='A protocol change is unresolved. Download progress and resolve it before freezing.';return;}download('audit-decisions.json');};
$('restore').onchange=async()=>{try{const value=JSON.parse(await $('restore').files[0].text());for(const k of ['schema_version','archive_sha256','sample_sha256','policy_sha256','audit_id','corpus_kind'])if(value[k]!==payload.template[k])throw Error();if(value.rows.length!==sample.length||value.rows.some((r,i)=>r.source_key!==sample[i].source_key))throw Error();decisions=value;$('reviewer').value=value.reviewer;$('date').value=value.reviewed_at;$('change').checked=value.protocol_change_requested===true;$('changeReason').value=value.protocol_change_reason;$('attest').checked=false;index=Math.max(0,value.rows.findIndex(r=>!complete(r)));draw();$('message').textContent='Progress restored.';}catch{$('message').textContent='Cannot restore: wrong sample or invalid progress file.'}};
$('date').value=new Date().toLocaleDateString('en-CA');draw();
</script></html>'''
