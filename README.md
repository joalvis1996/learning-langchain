# 러닝 랭체인

한빛미디어의 러닝 랭체인 실습 코드 저장소입니다.

원서 저장소는 [https://github.com/langchain-ai/learning-langchain](https://github.com/langchain-ai/learning-langchain)입니다.

<img src="./cover.png" width="300px">

## 설치 방법

이 저장소의 코드를 실행하기 위해서는 파이썬이나 자바스크립트 환경이 필요합니다.

**파이썬**

1.  필요한 경우 `.python-version` 파일에 명시된 Python 버전을 설치합니다. (예: pyenv 사용)
2.  가상 환경을 생성하고 활성화합니다. (권장)
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # macOS/Linux
    # .venv\\Scripts\\activate  # Windows
    ```
3.  필요한 라이브러리를 설치합니다.
    ```bash
    pip install -r requirements.txt
    ```

**자바스크립트**

1.  Node.js 및 npm (또는 yarn)을 설치합니다.
2.  프로젝트 루트에서 의존성을 설치합니다.
    ```bash
    npm install
    # 또는 yarn install
    ```
3.  환경 변수 설정이 필요한 경우 `.env.example` 파일을 복사하여 `.env` 파일을 만들고 내용을 채웁니다.


## 사용법

각 실습 코드는 해당 디렉토리 내에서 실행할 수 있습니다.

**파이썬 예제**

```bash
export OPENAI_API_KEY=<오픈AI-API-키>
cd python/ch00  # ch00는 해당 챕터 번호
python example_script.py
```

**자바스크립트 예제**

```bash
export OPENAI_API_KEY=<오픈AI-API-키>
cd javascript/ch00 # ch00는 해당 챕터 번호
# 실행 명령어 (예시)
node 00.xxxxx.js
```

자세한 내용은 도서와 소스 코드를 참고하세요. 