#!/bin/bash
# 크롤 완료 대기 후 PDF 배치 파싱 자동 시작
CRAWL_PID=52319
LOG="logs/auto_pipeline.log"

echo "$(date '+%H:%M:%S') 크롤 PID=$CRAWL_PID 완료 대기 중..." | tee -a $LOG

while kill -0 $CRAWL_PID 2>/dev/null; do
    sleep 30
done

echo "$(date '+%H:%M:%S') 크롤 완료! PDF 배치 파싱 시작..." | tee -a $LOG

PDF_COUNT=$(ls data/arxiv/pdfs/*.pdf 2>/dev/null | wc -l)
echo "$(date '+%H:%M:%S') 대상 PDF: $PDF_COUNT 건" | tee -a $LOG

python3 -m pipeline.batch_parse_pdfs --workers 4 >> $LOG 2>&1
echo "$(date '+%H:%M:%S') PDF 배치 파싱 완료!" | tee -a $LOG
