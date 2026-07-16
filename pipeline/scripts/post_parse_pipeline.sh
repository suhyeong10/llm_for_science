#!/bin/bash
# 파싱 완료 후 자동 실행 파이프라인
# 1) pipeline.batch_parse_pdfs 완료 대기
# 2) 인덱스 갱신
# 3) dedup 실행
# 4) eval 실행

set -e
cd "$(dirname "$0")/../.."

LOG="logs/post_parse_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== [$(date)] 파이프라인 시작 ==="

# 1) PDF 파싱 완료 대기
echo "[1/4] PDF 파싱 완료 대기 중..."
while pgrep -f "pipeline.batch_parse_pdfs" > /dev/null 2>&1; do
    LAST=$(tail -1 logs/batch_parse_20260408b.log 2>/dev/null || echo "대기중...")
    echo "  $(date +%H:%M:%S) 파싱 진행 중... $LAST"
    sleep 300  # 5분마다 확인
done
echo "[1/4] 파싱 완료!"

# 2) 인덱스 갱신
echo ""
echo "[2/4] DuckDB 인덱스 갱신..."
python3 main.py index
echo "[2/4] 인덱스 갱신 완료!"

# 3) 최종 통계
echo ""
echo "=== 파싱 후 통계 ==="
python3 -c "
import duckdb
con = duckdb.connect('data/index.db')
con.sql('SELECT full_text_source_type, COUNT(*) as cnt FROM papers GROUP BY 1 ORDER BY 2 DESC').show()
con.sql('SELECT COUNT(*) as total FROM papers').show()
"

# 4) dedup 실행
echo ""
echo "[3/4] 중복 제거 (dedup) 실행..."
python3 main.py dedup
echo "[3/4] dedup 완료!"

# 5) eval 실행
echo ""
echo "[4/4] 품질 평가 (eval) 실행..."
python3 main.py eval
echo "[4/4] eval 완료!"

# 최종 통계
echo ""
echo "=== 최종 통계 ==="
python3 main.py stats

echo ""
echo "=== [$(date)] 파이프라인 완료 ==="
