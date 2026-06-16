import os
import argparse
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM
import deepspeed
import torch.distributed as dist

from dataset import PretrainDataset

def parse_args():
    """개발자가 CLI(터미널) 명령어로 인프라와 자원을 수동 제어할 수 있도록 인자를 정의합니다."""
    parser = argparse.ArgumentParser(description="Manual Control DeepSpeed Script")
    
    # 1. 인프라 자원 및 설정 파일 컨트롤
    parser.add_argument("--ds_config", type=str, default="config/ds_config.json", 
                        help="적용할 DeepSpeed JSON 설정 파일 경로")
    
    # 2. 모델 및 학습 하이퍼파라미터 컨트롤
    parser.add_argument("--model_name_or_path", type=str, default="gpt2", 
                        help="HuggingFace 모델 경로 또는 로컬 모델 체크포인트 경로")
    parser.add_argument("--epochs", type=int, default=2, 
                        help="총 학습 에폭 수")
    parser.add_argument("--output_dir", type=str, default="output/checkpoints", 
                        help="최종 체크포인트를 저장할 디렉토리 경로")
    
    # DeepSpeed 가 구동 환경(Local Rank)을 파싱하기 위해 내부적으로 사용하는 인자
    parser.add_argument("--local_rank", type=int, default=-1, help="Internal DeepSpeed argument")
    
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. 분산 가속 환경을 위한 로컬 랭크(GPU 번호) 가져오기
    # 주입된 환경 변수 또는 아규먼트를 바탕으로 현재 프로세스가 담당할 GPU를 명확히 지정합니다.
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()

    # 2. 모델 설정 및 생성
    # 개발자가 명령어로 지정한 --model_name_or_path 에 맞춰 모델을 생성합니다.
    model_config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_config(model_config)

    # 3. 데이터셋 및 데이터로더 준비
    # 개발자가 수정한 ds_config.json 내부의 train_micro_batch_size_per_gpu 값을 수동으로 읽어와 연동합니다.
    with open(args.ds_config, "r") as f:
        ds_dict = json.load(f)
    batch_size = ds_dict.get("train_micro_batch_size_per_gpu", 1)

    dataset = PretrainDataset(block_size=512)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=deepspeed.comm.get_world_size(), rank=deepspeed.comm.get_rank(), shuffle=True
    )
    # 수동 파싱한 배치 사이즈를 데이터로더에 직접 제어 주입
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    # 4. DeepSpeed 엔진 초기화 (모델, 옵티마이저 바인딩)
    # 개발자가 명령어로 넘겨준 JSON 설정 파일 경로를 그대로 주입하여 세팅을 통제합니다.
    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        config=args.ds_config
    )

    # 전체 클러스터(멀티 노드 포함)에서 진짜 대장 GPU 딱 1대만 정보를 출력하도록 제어합니다.
    if dist.is_initialized() and dist.get_rank() == 0:
        world_size = dist.get_world_size()
        grad_acc = ds_dict.get("gradient_accumulation_steps", 1)
        print(f"==================================================")
        print(f"[활성화된 총 GPU 개수]: {world_size}")
        print(f"[GPU당 할당된 micro 배치 크기]: {batch_size}")
        print(f"[그라디언트 누적 스텝]: {grad_acc}")
        print(f"[최종 실질 배치 사이즈 (Total Batch Size)]: {world_size * batch_size * grad_acc}")
        print(f"==================================================")

    # 5. 훈련 루프 실행
    model_engine.train()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(model_engine.local_rank)
            labels = batch["labels"].to(model_engine.local_rank)

            # 순전파 (Forward)
            outputs = model_engine(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            # 역전파 (Backward)
            model_engine.backward(loss)
            
            # 옵티마이저 스텝 (Gradient Accumulation 반영하여 내부적으로 자동 조절됨)
            model_engine.step()

            # 전체 분산 서버를 통틀어 진짜 0번 마스터 GPU 프로세스에서만 로그 출력 (터미널 도배 방지)
            if dist.is_initialized() and dist.get_rank() == 0 and step % 5 == 0:
                print(f"Epoch: {epoch} | Step: {step} | Loss: {loss.item():.4f}")

    # 6. 테스트 완료 후 최종 체크포인트 안전 저장
    os.makedirs(args.output_dir, exist_ok=True)
    
    # save_checkpoint는 분산 저장을 위해 모든 GPU가 수동으로 같이 호출해 주어야 합니다.
    model_engine.save_checkpoint(args.output_dir, tag="final_run")
    
    # 완료 메시지는 마스터 프로세스에서만 깔끔하게 한 번 출력합니다.
    if dist.is_initialized() and dist.get_rank() == 0:
        print(f"🚀 학습 완료 및 '{args.output_dir}'에 체크포인트 저장 성공!")

if __name__ == "__main__":
    main()
