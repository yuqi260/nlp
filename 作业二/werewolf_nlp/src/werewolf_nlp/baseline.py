"""Direct baseline: public snapshot in, nine role predictions out."""
import json
from collections import Counter
from itertools import combinations, permutations
from urllib import request, error

ROLES = ('seer', 'witch', 'hunter', 'werewolf', 'villager')
COUNTS = {'seer': 1, 'witch': 1, 'hunter': 1, 'werewolf': 3, 'villager': 3}
PROMPT_VERSION = 'direct_v1'
SYSTEM = '''你是狼人杀对局的旁观者。仅根据给出的公开规则与事件判断九名玩家的具体身份。
玩家的发言可能为假、误判或伪装，不是已确认事实。不要使用任何未公开或未来信息。
角色配置：seer 预言家1人、witch 女巫1人、hunter 猎人1人、werewolf 狼人3人、villager 平民3人。
只输出JSON：{"assignments":[{"player_id":1,"role":"seer"}, ...]}，玩家1至9各出现一次；role必须为上述英文值。'''

RULES = """本局规则与记录说明：
- 预言家每晚可私下查验一名玩家是否为狼人。发言中自报的身份、查验结果和夜间行动均是玩家声明，推断时应结合其他发言与公开行为核对。
- 女巫有一次解药和一次毒药，同一晚不能同时用药。解药用于救当夜被狼人袭击的玩家，毒药可淘汰一人。平安夜或死亡名单只给出公开结果，具体死因和用药情况仍需推断。
- 猎人被狼人杀害或被投票放逐后，可选择翻牌带走一名存活玩家；被毒杀时不能发动。实际 shoot 事件中的 speaker 是翻牌开枪的猎人，target 是被带走的玩家，目标身份仍需推断。没有开枪记录时，出局者仍可能是猎人。
- vote_results 记录实际票型。votes 中 player 为投票者，target 为投票目标，空值表示未投给任何编号。唯一最高票者被放逐；tied_players 非空表示该次投票尚未决定出局，应结合后续复投。发言中的归票和投票承诺是声明。"""
PROMPTS = {
    PROMPT_VERSION: SYSTEM,
    'rules_v1': SYSTEM + '\n\n' + RULES,
    'evidence_index_v1': SYSTEM + '\n\n' + RULES + '\n\n玩家发言索引按轮次列出各玩家发言的事件编号；请结合原始事件、公开票型和结算进行判断。',
}
EVIDENCE_FORMAT = '''请先按发言顺序整理身份自称和查验声明，再输出身份预测。每条声明保留原文，标明事件编号与发言者。只摘录发言者明确作出的自称或自己的查验声明，无法确定时跳过。玩家声明可能为假，不视为事实。
只输出 JSON，格式为 {"claims":[{"event_id":"事件编号","speaker":1,"kind":"role_claim","quote":"发言中的连续原文"}],"assignments":[{"player_id":1,"role":"seer"}]}。kind 只能为 role_claim（自称身份）或 inspection_claim（声称自己的查验结果）。assignments 中玩家1至9各出现一次，role为规定的英文值。'''
PROMPTS['claims_v1'] = SYSTEM[:SYSTEM.index('只输出JSON')] + RULES + '\n\n' + EVIDENCE_FORMAT
PROMPTS['hypotheses_v1'] = SYSTEM[:SYSTEM.index('只输出JSON')] + '''先列出自称预言家的玩家及各自声称的查验结果，结合发言、实际投票和公开结算比较各假设的支持与反对证据。身份声明和查验结果可能是谎言，玩家声称的投票意向可能与实际票型不同。公开死亡名单不能单独确定死因。若存在实际开枪，开枪者是猎人。综合历史证据进行角色分配。
只输出JSON，格式为 {"hypotheses":[{"seer_candidate":1,"support":"简要支持证据及事件编号","opposition":"简要反对证据及事件编号"}],"assignments":[{"player_id":1,"role":"seer"}]}。如果没有预言家自称，hypotheses 可为空。玩家1至9各出现一次；role必须为规定的英文值。'''
PROMPTS['hypotheses_v2'] = PROMPTS['hypotheses_v1'] + '''
仅引用提供的事件。投票出局和死亡名单不能证实身份、死因或获胜阵营，玩家的用药和查验说法也不能直接证实。事件没有公布胜负时不得声称胜负已确定。推断要标为可能性，支持与反对均须引用对应事件编号；没有证据就写信息不足。'''
PROMPTS['scores_v1'] = SYSTEM[:SYSTEM.index('只输出JSON')] + '''对每名玩家的每种候选身份分别评分，整数范围 -2 到 2，表示综合公开证据后与该身份的匹配程度，0 表示信息不足。发言和查验结果是声明，可能为假。实际开枪者是猎人。分别评估各候选角色，后续程序会根据角色数量进行整体分配。
只输出 JSON：{"scores":[{"player_id":1,"seer":0,"witch":0,"hunter":0,"werewolf":0,"villager":0}]}。玩家1至9各出现一次，五个身份均须评分。'''
PROMPTS['scores_float_v1'] = SYSTEM[:SYSTEM.index('只输出JSON')] + '''对每名玩家的五种候选身份分别给出 0 到 100 的数值匹配分，可保留一位小数。分数表达相对匹配程度，不要求各角色之和为100，不视为概率。发言和查验结果只是声明，可能为假。实际开枪者是猎人。后续程序根据角色数量整体分配。
只输出 JSON：{"scores":[{"player_id":1,"seer":50.0,"witch":50.0,"hunter":50.0,"werewolf":50.0,"villager":50.0}]}。玩家1至9各出现一次，五个身份均须评分。'''
PROMPTS['behavior_v1'] = SYSTEM + '\n附带公开行为索引，仅用于定位原始事件；复投变化不自动证明阵营。'
PROMPTS['atom_index_v1'] = SYSTEM + '\n附带机器提取的声明候选索引，仅供定位原文；索引可能遗漏或误读语气、指代和立场。声明不是事实，请以原始公开事件为准。'
PROMPTS['atom_links_v1'] = PROMPTS['atom_index_v1'] + '\n另附同一玩家在投票之前对该目标的立场候选与实际投票的关联；立场可能有误或后来改变，不自动证明阵营或动机。'
PROMPTS['wolfgame_direct_v1'] = SYSTEM + '\n这份记录有警长竞选和警长票权（1.5票），player_id 是本次分析代号；原游戏的公开座位号由 seat_ref 显示，原文也可能按名字指代。输出必须使用 player_id。公开的猎人翻牌事件可确认该玩家为猎人。不能从开枪目标、出局或玩家声明确定身份。'


def make_messages(snapshot, prompt_version=PROMPT_VERSION):
    public = []
    for e in snapshot['public_events']:
        if e['game_id'] != snapshot['game_id'] or e['order'] > snapshot['cutoff_order']:
            raise ValueError('event outside snapshot')
        allowed = {'event_id', 'day', 'order', 'type', 'speaker', 'text', 'votes', 'eliminated_player', 'tied_players', 'target'}
        public.append({k: v for k, v in e.items() if k in allowed})
    payload = {'day': snapshot['day'], 'events': public}
    if prompt_version == 'wolfgame_direct_v1':
        directory = snapshot.get('player_directory')
        if not isinstance(directory, list) or len(directory) != 9:
            raise ValueError('missing public player directory')
        payload['player_directory'] = [{'player_id': item['player_id'], 'name': item['name'], 'seat_ref': item['seat_ref']} for item in directory]
    if prompt_version == 'behavior_v1':
        from .behavior import extract_behavior
        payload['behavior_index'] = extract_behavior(snapshot)
    if prompt_version == 'evidence_index_v1':
        index = {}
        for event in public:
            if event['type'] == 'speech':
                day = str(event['day'])
                player = str(event['speaker'])
                index.setdefault(day, {}).setdefault(player, []).append(event['event_id'])
        payload['speech_index_by_day_and_player'] = index
    return [{'role': 'system', 'content': PROMPTS[prompt_version]}, {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]


def parse_response(text):
    text = text.strip()
    if text.startswith('```') and text.endswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    value = json.loads(text)
    items = value.get('assignments') if isinstance(value, dict) else None
    if not isinstance(items, list) or len(items) != 9:
        raise ValueError('assignments must contain nine players')
    result = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError('invalid assignment')
        player, role = item.get('player_id'), item.get('role')
        if type(player) is not int or player not in range(1, 10) or player in result or role not in ROLES:
            raise ValueError('invalid player or role')
        result[player] = role
    return result, dict(Counter(result.values())) == COUNTS


def validate_claims(text, snapshot):
    value = json.loads(text.strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip())
    claims = value.get('claims') if isinstance(value, dict) else None
    if not isinstance(claims, list):
        raise ValueError('claims must be a list')
    events = {event['event_id']: event for event in snapshot['public_events'] if event['type'] == 'speech'}
    for claim in claims:
        if not isinstance(claim, dict):
            raise ValueError('claim must be an object')
        event = events.get(claim.get('event_id'))
        if not event or type(claim.get('speaker')) is not int or claim['speaker'] != event['speaker']:
            raise ValueError('claim references invalid event or speaker')
        if claim.get('kind') not in ('role_claim', 'inspection_claim'):
            raise ValueError('invalid claim kind')
        quote = claim.get('quote')
        if not isinstance(quote, str) or not quote.strip() or quote not in event['text']:
            raise ValueError('claim quote must appear in referenced speech')
    return claims


def assign_scores(text, snapshot):
    text = text.strip()
    if text.startswith('```') and text.endswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    value = json.loads(text)
    rows = value.get('scores') if isinstance(value, dict) else None
    if not isinstance(rows, list) or len(rows) != 9:
        raise ValueError('scores must contain nine players')
    scores = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('invalid score row')
        player = row.get('player_id')
        if type(player) is not int or player not in range(1, 10) or player in scores:
            raise ValueError('invalid score player')
        if any(type(row.get(role)) is not int or row[role] not in range(-2, 3) for role in ROLES):
            raise ValueError('scores must be integers from -2 to 2')
        scores[player] = {role: row[role] for role in ROLES}
    shooters = {event['speaker'] for event in snapshot['public_events'] if event['type'] == 'shoot'}
    if len(shooters) > 1:
        raise ValueError('multiple public shooters')
    best, best_score, ties = None, None, 0
    players = range(1, 10)
    for seer, witch, hunter in permutations(players, 3):
        if shooters and hunter not in shooters:
            continue
        remaining = [player for player in players if player not in (seer, witch, hunter)]
        for wolves in combinations(remaining, 3):
            roles = {player: 'villager' for player in remaining}
            roles.update({player: 'werewolf' for player in wolves})
            roles.update({seer: 'seer', witch: 'witch', hunter: 'hunter'})
            total = sum(scores[player][role] for player, role in roles.items())
            if best_score is None or total > best_score:
                best, best_score, ties = roles, total, 1
            elif total == best_score:
                ties += 1
    return best, scores, best_score, ties


def call_api(base_url, api_key, model, messages, temperature, max_tokens, timeout, thinking=None):
    body = {'model': model, 'messages': messages, 'temperature': temperature, 'max_tokens': max_tokens, 'stream': False}
    if thinking is not None:
        body['enable_thinking'] = thinking
    payload = json.dumps(body, ensure_ascii=False).encode('utf-8')
    req = request.Request(base_url.rstrip('/') + '/chat/completions', data=payload, headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + api_key}, method='POST')
    try:
        with request.urlopen(req, timeout=timeout) as response:
            result = json.load(response)
    except error.HTTPError as exc:
        # Never log service error bodies; they may echo sensitive request data.
        raise RuntimeError(f'HTTP {exc.code}') from None
    content = result['choices'][0]['message'].get('content')
    if not isinstance(content, str) or not content.strip():
        raise ValueError('empty model output')
    return content, result.get('usage'), result.get('model'), result['choices'][0].get('finish_reason')
