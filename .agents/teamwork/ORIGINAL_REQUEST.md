# Original User Request

## 2026-09-26T15:25:26Z

# Teamwork Project Prompt

메이플스토리 클라이언트 데이터(`C:\mp`의 `BossSuu.img.json`, `bossSuu.img.json` 맵 아틀라스, `wz_json_restorer.py`)를 기반으로, 스우 1페이즈(중앙 회전 십자 레이저 및 수직 낙하물 탄막) 기믹을 Gymnax 규격의 고속 JAX/XLA 함수형 강화학습 환경으로 구축하고, C:\mp WZ 파싱 파이프라인, PureJaxRL/Stoix 연동 래퍼, 그리고 하드웨어(Ryzen 5600X / RX 6600 XT / 48GB RAM) 맞춤형 SPS 벤치마크 스위트를 구현합니다.

Working directory: c:\Users\ROCmAdmin\Documents\antigravity\bold-faraday
Integrity mode: development

## Hardware Profile & Recommended Scale
- CPU: AMD Ryzen 5 5600X (6 Cores / 12 Logical Processors, 32MB L3 Cache)
- RAM: 48 GB
- GPU: AMD Radeon RX 6600 XT (8GB VRAM) / WSL2 Ubuntu-24.04-ROCmLab 지원
- 권장 시뮬레이션 배치 규모:
  - CPU 테스트: 1,024 ~ 2,048 병렬 환경 (L3 캐시 최적화 및 50k~200k SPS 달성)
  - GPU/ROCm 가속: 4,096 ~ 8,192 병렬 환경 (RX 6600 XT 32 CU 포화 및 500k~2M+ SPS 달성)

## Requirements

### R1. Gymnax 기반 스우 1페이즈 시뮬레이터 코어 (`src/maple_gymnax/envs/lotus_phase1.py`)
- `flax.struct.dataclass` 기반으로 불변 환경 매개변수(`EnvParams`)와 동적 틱 상태(`EnvState`)를 완벽히 분리.
- `C:\mp` 클라이언트 물리 제원 반영:
  - 맵 좌표계 (1366 x 768), 코어 중심점 (683.0, 384.0), 바닥 y좌표 ~605.0.
  - 플레이어 히트박스 (40.0 x 60.0), 이동 속도 400.0 px/s, 기준 dt = 1/60s.
  - 회전 십자 레이저: 각속도 0.5235 rad/s, 점-직선 직교 거리 공식 + 방향성 내적(Dot product) 마스킹.
  - 수직 낙하물: 최대 30개 정적 패딩 배열(`Static Padded Array`) 및 불리언 활성 마스크(`Boolean Mask`) 패턴, 유클리디안 거리(`jnp.linalg.norm`) 기반 벡터화 충돌 판정.
- XLA JIT 무분기(Branch-free) 연산: Python `if/else` 및 조건문 분기를 철저히 배제하고 `jnp.where`, `jax.lax.cond/select` 연산자로 상태 전이 구현 (`ConcretizationTypeError` 원천 방지).

### R2. C:\mp 클라이언트 데이터 파서 파이프라인 (`src/maple_gymnax/parser/wz_parser.py`)
- `C:\mp\wz_json_restorer.py`의 WZ 복원 알고리즘을 확장하여, `C:\mp\Restored_Data\Mob\BossPattern`(`BossSuu.img.json`) 및 `Map\Back`(`bossSuu.img.json`) 데이터를 파싱.
- 픽셀 앵커 오프셋, 프레임 delay(ms -> sec), 히트박스 치수, 스킬 데미지를 구조화된 `EnvParams` JSON 스키마로 자동 추출/변환하는 파이프라인 구현.

### R3. 하위 강화학습 프레임워크 연동 및 롤아웃 러너 (`src/maple_gymnax/wrappers/`)
- `FlattenObservationWrapper`: PureJaxRL 신경망 입력을 위한 다차원 관측치(플레이어 좌표, 체력, 레이저 각도, 낙하물 좌표/마스크) 1차원 평탄화.
- `jax.vmap` + `jax.lax.scan` 기반의 에피소드 고속 롤아웃 러너.
- Stoix/Stoa 어댑터 호환 인터페이스 (`AutoResetWrapper`, 에피소드 메트릭 기록기 지원).
- Flashbax Pytree 경험 리플레이 버퍼 연동 무복사(Zero-copy) 인터페이스.

### R4. 에이전트 페르소나 제어 프롬프트 (`src/maple_gymnax/prompts/persona_system_prompt.py`)
- LLM이 비정형 클라이언트 데이터를 수학적 텐서 물리량으로 변환하고 Gymnax 규격의 Mock 코드를 안정적으로 생성하도록 제어하는 마스터 시스템 프롬프트 및 템플릿 문서화.

## Acceptance Criteria

### Automated Verification
- [ ] `uv venv --python 3.12` 가상환경에서 `jax`, `flax`, `gymnax`, `pytest` 의존성이 정상 설치되고 로드되어야 함.
- [ ] `tests/test_lotus_phase1.py`에서 `jax.jit(env.step)` 및 `jax.jit(env.reset)` XLA 컴파일 에러가 발생하지 않아야 함.
- [ ] `jax.vmap`을 통한 1,024 ~ 4,096개 병렬 환경 시뮬레이션에서 Shape Mismatch 없이 정상 동작해야 함.
- [ ] 레이저 직교 거리 판정, 방향성 마스킹, 낙하물 거리 판정, 데미지 계산 및 에피소드 종료 조건 단위 테스트 100% 통과.
- [ ] `tests/test_wz_parser.py`에서 `C:\mp` 실측 데이터의 JSON 스키마 변환 및 `EnvParams` 생성 검증 통과.
- [ ] `benchmarks/benchmark_sps.py` 실행 시 다양한 배치 크기(256, 512, 1024, 2048, 4096)에서 초당 스텝 수(SPS) 벤치마크가 정상 측정되어야 함.

## 2026-09-26T15:35:16Z

[CRITICAL DOMAIN KNOWLEDGE UPDATE - REMASTERED LOTUS (스우 리마스터/리메이크)]

사용자의 명시적 지침 및 로컬 Docker Firecrawl(port 8085) 딥리서치 결과, 메이플스토리 스우는 2024년 4월 공식 리마스터(리메이크)되었으며, C:\mp의 `BossSuu.img.json`에 기록된 1001~1009 및 destruction/overload 데이터가 바로 이 리마스터 스우의 패턴입니다.

### 리마스터 스우 1페이즈 핵심 기믹 및 물리 수학 사양:
1. **보안/섬멸 게이지 (Security & Annihilation Gauge)**:
   - 자연 증가: 초당 0.6% (노멀), 0.8% (하드), 2.0% (익스트림).
   - 100% 도달 시 **'섬멸 모드(Overload/Destruction)'** 진입 (25초간 지속).
   - 섬멸 모드 중 즉사기 **'가로 포격 / 침입자 격퇴 프로토콜(1006-000)'** (피격 시 0.5s마다 100% HP 피해, 우측 안전지대) 및 **'전기장 설치(1006-002)'** (4초간 0.36s마다 5% 피해, 5틱 누적 시 강제 점프+스턴) 발생.
2. **공멸형 패턴 및 보스 유도(Friendly Fire) 메커니즘**:
   - **추적 레이저 (1001-000)**: 상단 기계팔 2줄 레이저 (1.0s 플레이어 추적 조준 후 정지, 1.0s 후 발사).
     - 플레이어 피격 시: HP 15% 데미지 + 게이지 +7~20% 상승.
     - **스우 본체 유도 적중 시: 게이지 -7~20% 감소 + 스우 보호막(Shield) 대폭 파괴.**
   - **소형 기계팔 돌진 (1001-001)**: 12회 연속 내리찍기 (0.8s 간격, 피격 시 5% 데미지).
     - 플레이어 피격 시: 게이지 +3% 상승.
     - **스우 본체 유도 적중 시: 게이지 -3% 감소.**
3. **바닥 전류 방출 (즉사기 / 점프 회피)**:
   - 바닥에 푸른 전류 흐른 후 폭발 -> 더블 점프 또는 체공으로 회피.
4. **보호막(Shield) 생성**:
   - 스우 본체에 보호막 생성 -> 시간 내 미파괴 시 체력 회복 -> 추적 레이저를 스우에게 맞추면 파괴.

이 리메이크 사양을 `maple_gymnax.envs.lotus_phase1`의 상태 공간(`EnvState`), 매개변수(`EnvParams`), 상태전이(`step_env`), 보상 함수(게이지 관리 유도 보상, 레이저 유도 보상)에 최우선으로 반영하고, 구버전 십자 레이저 기믹과 리마스터 게이지/공멸 기믹을 유연하게 스위칭 또는 결합할 수 있도록 모듈화해 주십시오.
