"""English-task policy shared by construction, export and frozen snapshots."""
import re

_CJK = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7a3]')


def english_task(record):
    meta = record.get('meta') or {}
    dataset = record.get('dataset') or meta.get('dataset') or ''
    if dataset == 'maskedit_cn':
        return False
    language = record.get('language') or meta.get('language')
    if language and str(language).lower() not in ('en', 'eng', 'english', 'en-us', 'en-gb'):
        return False
    instruction = meta.get('instruction') or record.get('instruction')
    return isinstance(instruction, str) and bool(instruction.strip()) and not _CJK.search(instruction)


def english_supervision(record):
    import json
    reasoning = record.get('reasoning')
    if reasoning is None and record.get('answer'):
        reasoning = json.loads(record['answer'])
    content = {'reasoning':reasoning,'code':record.get('code','')}
    return not _CJK.search(json.dumps(content,ensure_ascii=False))
