"""Outcome reward gates for semantically meaningful image edits."""
import math


def score_judgment(result):
    result = dict(result)
    names = ('instruction_adherence', 'preservation', 'visual_quality')
    scores = [float(result[k]) for k in names]
    if not all(math.isfinite(v) and 1 <= v <= 5 for v in scores):
        raise ValueError('Judge scores must be finite numbers from 1 to 5.')
    raw = sum(w * (v - 1) / 4 for w, v in zip((.7, .2, .1), scores))
    eligible = scores[0] >= 3 and scores[1] >= 3 and scores[2] >= 2
    result.update(raw_reward=raw, reward=raw if eligible else 0.,
                  reward_eligible=eligible, reward_version='semantic_gates_v2')
    result['pass'] = (result.get('pass') is True and scores[0] >= 4
                      and scores[1] >= 4 and scores[2] >= 3)
    if result.get('rubric_version')=='full_visible_target_v1':
        extent=float(result['target_extent_score'])
        if not math.isfinite(extent)or not 1<=extent<=5 or not isinstance(result.get('target_extent_complete'),bool):
            raise ValueError('Strict judging requires valid target extent fields.')
        result['pass']=result['pass']and result['target_extent_complete']and extent>=4
        result['reward_eligible']=result['reward_eligible']and extent>=3
        if not result['reward_eligible']:result['reward']=0.
        result['reward_version']='semantic_gates_v3_extent'
    return result
