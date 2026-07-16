# Upstream PR Workflow (쉽게 정리)

## 왜 헷갈리나?

PR에는 두 축이 있습니다.

- `base`: 변경을 **받는** 쪽 브랜치
- `head`: 변경을 **보내는** 쪽 브랜치

즉, PR은 항상 `head -> base` 방향입니다.

## 이번 케이스에 맞춰 보기

- 내 작업 저장소(fork): `eepLearning/llm_for_science`
- 반영 대상 저장소(upstream): `suhyeong10/llm_for_science`

일반적으로 upstream에 반영할 때는:

1. 내 fork에서 작업 브랜치를 만든다 (`feat/...`)
2. 작업 내용을 커밋/푸시한다
3. upstream 저장소를 대상으로 PR을 만든다
   - `base = upstream의 반영 브랜치` (예: `main` 또는 `feat/...`)
   - `head = eepLearning:내 작업 브랜치`

## head를 main으로 잡으면 왜 안 좋은가?

`head=main`은 보통 "작업 브랜치"가 아니라 "기본 브랜치"라서,
변경 분리가 안 되고 PR 의도가 흐려집니다.

권장:

- `head`: 작업 브랜치 (`feat/llm-cpt-full-upstream` 같은 이름)
- `base`: 업스트림 수용 브랜치 (`main` 또는 `feat/llm-cpt-full`)

## 브랜치 전략 (실무 권장)

1. `main`은 항상 안정 상태 유지
2. 기능 단위로 `feat/...` 브랜치 생성
3. upstream 반영용은 이름을 명확히 분리
   - 예: `feat/llm-cpt-full-upstream`
4. 리뷰 중에는 `rebase origin/main`으로 최신화
5. 충돌 해결 후 `git push --force-with-lease`

## 실제 명령 예시

```bash
# 1) main 최신화
git checkout main
git pull origin main

# 2) 작업 브랜치 생성
git checkout -b feat/llm-cpt-full-upstream

# 3) 변경 반영 후 푸시
git push -u origin feat/llm-cpt-full-upstream
```

PR 생성 시:

- Base repository: `suhyeong10/llm_for_science`
- Base branch: `feat/llm-cpt-full` (또는 팀에서 지정한 수용 브랜치)
- Head repository: `eepLearning/llm_for_science`
- Compare branch: `feat/llm-cpt-full-upstream`

## 체크리스트

- 산출물 파일이 커밋에 없는가?
- README/설정/코드 변경 범위가 PR 설명과 일치하는가?
- 충돌 해결 후 로컬 테스트(최소 스모크)를 다시 돌렸는가?
