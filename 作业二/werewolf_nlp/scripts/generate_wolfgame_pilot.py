"""Generate one-day synthetic WolfGame pilot using an in-memory credential.

Run with WolfGame's isolated .venv Python. Default is dry run. Agent-generated
games are not human games and require separate rule and leakage audits.
"""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'data/raw/wolfgame_candidate'
sys.path.insert(0, str(ROOT / 'src'))
from werewolf_nlp.local_config import load_local_config


def main():
    load_local_config(ROOT / '.env')
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true', help='make paid model calls')
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--run-name', default='runs_llm_pilot_v4')
    args = parser.parse_args()
    if not args.run_name.isidentifier() or not args.run_name.startswith('runs_llm_pilot_'):
        parser.error('run-name must be an identifier starting with runs_llm_pilot_')
    if (ENGINE / args.run_name).exists() or (ENGINE / (args.run_name + '_agents')).exists():
        parser.error('run or agent directory already exists')
    if not args.execute:
        print('Dry run: one synthetic day, Qwen3.5-27B, no training or API calls.')
        return
    key = os.environ.get('WEREWOLF_API_KEY')
    if not key:
        parser.error('WEREWOLF_API_KEY is required')
    os.environ['LANGSMITH_TRACING'] = 'false'
    os.chdir(ENGINE)
    sys.path.insert(0, str(ENGINE / 'src'))
    import wolfgame.config as config
    settings = {'WOLFGAME_API_KEY': key, 'WOLFGAME_MODEL': 'qwen3.5-27b',
                'WOLFGAME_MODEL_PROFILE': 'qwen3.5-27b-local',
                'WOLFGAME_MODEL_PROFILES_FILE': str(ENGINE / 'config/model_profiles_qwen35_local.json'),
                'WOLFGAME_MAX_CONCURRENCY': '4', 'WOLFGAME_EXPLICIT_CACHE': 'false'}
    settings.update({f'WOLFGAME_TASK_{task.upper()}_THINKING': 'false' for task in config.DEFAULT_LLM_TASK_PROFILES})
    config.dotenv_values = lambda *unused_args, **unused_kwargs: settings
    from wolfgame.cli import main as wolfgame_main
    sys.argv = ['wolfgame', 'play', '--provider', 'compatible', '--env-file', 'nonexistent-ephemeral.env',
                '--model', 'qwen3.5-27b', '--model-profile', 'qwen3.5-27b-local',
                '--runs-dir', args.run_name, '--agents-dir', args.run_name + '_agents',
                '--seed', str(args.seed), '--max-days', '1', '--no-wolf-explosion',
                '--no-explicit-cache', '--public-only']
    wolfgame_main()


if __name__ == '__main__':
    main()
