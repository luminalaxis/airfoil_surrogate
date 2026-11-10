import sys

from .cli import main

if __name__ == "__main__":  # 스크리닝 병렬 처리(spawn) 시 자식 프로세스에서 다시 실행되지 않도록
    sys.exit(main())
