# qrl: build, test and training entry points (rl/README.md). Every recipe runs from this directory (rl/).
#
# The core package qrl/ needs only `-I .` (no QUMO). The bridge (qumo_bridge/) and its tools need a QUMO checkout
# at QUMO_ROOT and build with `-I $(QUMO_ROOT) -I .`; bridge tests run from QUMO_ROOT because they read QUMO data
# (models/, tests/fixtures/ai, bench/) by QUMO-relative paths. The QUMO root Makefile delegates here (make test-rl,
# test-rl-slow, train-ai, train-ai-smoke, models-verify) with absolute MOJO, AI_BUILD and MODELS.

QUMO_ROOT ?= ..
MOJO ?= $(QUMO_ROOT)/.venv/bin/mojo
# where the tools and runs go, and the model root the training targets export into / models-verify checks
AI_BUILD ?= build/ai_train
MODELS ?= $(QUMO_ROOT)/models/ai_transpiler

# MOJO and AI_BUILD as seen from QUMO_ROOT (the bridge tests run there): absolute unless MOJO is a bare command
MOJO_FROM_ROOT := $(if $(findstring /,$(MOJO)),$(abspath $(MOJO)),$(MOJO))
AI_BUILD_ABS := $(abspath $(AI_BUILD))

AI_CORES ?= 4
# two lanes share the same pool threads and each runs about 0.69x slower (docs/MOJO_PORT.md section 3, rule 8)
AI_LANES ?= 1
# hard wall-clock cap of the whole synthesis campaign (the shipped budgets total 22,200 s)
AI_SYNTH_BUDGET_S ?= 28800
# behaviour-cloning wall-clock budget per round (train_route stops after the epoch that crosses it)
AI_ROUTE_BC_BUDGET_S ?= 1800
AI_ROUTE_WORK ?= $(AI_BUILD)/route
AI_ROUTE_ID ?= route_mlp_h64_v1

CORE_TOOLS := train_synth export_synth train_all models_verify make_fixtures train_route
BRIDGE_TOOLS := route_tool export_route train_route_ppo eval_synth eval_blocks
AI_TOOLS := $(CORE_TOOLS) $(BRIDGE_TOOLS)

# fast core tests (every file but the statistical *_slow and the driver smoke test)
CORE_TESTS := $(filter-out %_slow.mojo tests/test_rl_smoke.mojo,$(wildcard tests/test_rl_*.mojo))
CORE_SLOW := $(wildcard tests/test_rl_*_slow.mojo) tests/test_rl_smoke.mojo
BRIDGE_TESTS := $(filter-out %_e2e.mojo,$(wildcard qumo_bridge/tests/test_bridge_*.mojo))
BRIDGE_SLOW := qumo_bridge/tests/test_bridge_e2e.mojo

QRL_SRC := $(shell find qrl -name '*.mojo' 2>/dev/null)
BRIDGE_SRC := $(shell find qumo_bridge -name '*.mojo' ! -path '*/tests/*' 2>/dev/null) $(shell find $(QUMO_ROOT)/qumo -name '*.mojo' 2>/dev/null)

.PHONY: all test test-slow test-bridge test-bridge-slow test-all check-independence tools core-tools smoke bench-rl \
	train-ai train-ai-synth train-ai-route train-ai-route-ppo train-ai-smoke models-verify check-pure-mojo

all: core-tools

# ---------------------------------------------------------------- tools
# one binary per main; core mains build with -I . only, bridge mains also see QUMO
$(AI_BUILD)/bin/%: drivers/synth/%.mojo $(QRL_SRC)
	@mkdir -p $(dir $@)
	$(MOJO) build -I . -o $@ $<
$(AI_BUILD)/bin/%: drivers/route/%.mojo $(QRL_SRC)
	@mkdir -p $(dir $@)
	$(MOJO) build -I . -o $@ $<
$(AI_BUILD)/bin/%: drivers/common/%.mojo $(QRL_SRC)
	@mkdir -p $(dir $@)
	$(MOJO) build -I . -o $@ $<
$(AI_BUILD)/bin/%: qumo_bridge/drivers/%.mojo $(QRL_SRC) $(BRIDGE_SRC)
	@mkdir -p $(dir $@)
	$(MOJO) build -I $(QUMO_ROOT) -I . -o $@ $<

core-tools: $(addprefix $(AI_BUILD)/bin/,$(CORE_TOOLS))  ## the QUMO-free tools (synthesis training / export / campaigns, router BC, models_verify)
tools: $(addprefix $(AI_BUILD)/bin/,$(AI_TOOLS))  ## every tool, bridge tools included (needs QUMO_ROOT)

# ---------------------------------------------------------------- tests
# one line per file: PASS / FAIL and the wall time (each fast file must stay under 30 s)
run_tests = fail=0; for t in $(1); do \
	  s=$$(date +%s); out=$$($(2) $$t 2>&1 | grep -v Crashpad | tail -1); e=$$(( $$(date +%s) - s )); \
	  case "$$out" in *OK*) echo "PASS $$t ($${e}s)";; *) echo "FAIL $$t: $$out"; fail=1;; esac; \
	done; exit $$fail

test: check-independence  ## core: the independence check and the fast qrl tests (only -I .; no QUMO needed)
	@$(call run_tests,$(CORE_TESTS),$(MOJO) run -I .)

test-slow:  ## core: statistical tests, the driver smoke test and test_rl_tensor with -D ASSERT=all
	@$(call run_tests,$(CORE_SLOW),$(MOJO) run -I .)
	@out=$$($(MOJO) run -D ASSERT=all -I . tests/test_rl_tensor.mojo 2>&1 | grep -v Crashpad | tail -1); \
	case "$$out" in *OK*) echo "PASS tests/test_rl_tensor.mojo (-D ASSERT=all)";; *) echo "FAIL tests/test_rl_tensor.mojo: $$out"; exit 1;; esac

# (this directory must be QUMO_ROOT/rl: the bridge tests name their files rl/...)
BRIDGE_ENV := export QRL_FIXTURES=rl/tests/fixtures QRL_MODEL_ROOT=models/ai_transpiler QUMO_E2E_BUILD=$(AI_BUILD_ABS) MOJO=$(MOJO_FROM_ROOT)

test-bridge:  ## bridge: fast qrl <-> QUMO tests (run from QUMO_ROOT with -I . -I rl)
	@cd $(QUMO_ROOT) && $(BRIDGE_ENV) && $(call run_tests,$(addprefix rl/,$(BRIDGE_TESTS)),$(MOJO_FROM_ROOT) run -I . -I rl)

test-bridge-slow: tools  ## bridge: the end-to-end test of the (re)built tools against QUMO's runtime (A7, A10; minutes)
	@cd $(QUMO_ROOT) && $(BRIDGE_ENV) && $(call run_tests,$(addprefix rl/,$(BRIDGE_SLOW)),$(MOJO_FROM_ROOT) run -I . -I rl)

test-all: test test-bridge test-slow test-bridge-slow  ## everything

bench-rl:  ## performance gates G1-G10 of docs/MOJO_PORT.md section 15, one JSON line each (run on an idle machine)
	@mkdir -p build
	$(MOJO) build -I . -o build/bench_rl bench/bench_rl.mojo && ./build/bench_rl

check-pure-mojo:  ## the no-Python / no-Rust scans of the training path (runs from QUMO_ROOT; check_pure_mojo.sh)
	cd $(QUMO_ROOT) && rl/check_pure_mojo.sh

check-independence:  ## fails if qrl/, drivers/ or tests/ import qumo.* (and compiles a core module with -I . only)
	@./check_independence.sh "$(MOJO)"

# ---------------------------------------------------------------- training (docs/TRAINING.md)
models-verify: $(AI_BUILD)/bin/models_verify  ## sha256 of every model file under $$QUMO_MODEL_DIR, else MODELS, against its INDEX.json
	@QUMO_MODEL_DIR=$${QUMO_MODEL_DIR:-$(MODELS)} $<

train-ai: train-ai-synth train-ai-route  ## retrain the synthesis models and the router into MODELS (hours)
	@$(MAKE) --no-print-directory models-verify

train-ai-synth: $(addprefix $(AI_BUILD)/bin/,train_all train_synth export_synth eval_synth)  ## synthesis models of drivers/synth/campaign.json (cap AI_SYNTH_BUDGET_S)
	@pids=; for k in $$(seq 1 $(AI_LANES)); do \
	  $(AI_BUILD)/bin/train_all --campaign drivers/synth/campaign.json --lane $$k/$(AI_LANES) --root $(MODELS)/synth \
	      --cores $(AI_CORES) --budget-cap-s $(AI_SYNTH_BUDGET_S) --work $(AI_BUILD)/synth & pids="$$pids $$!"; done; \
	  rc=0; for p in $$pids; do wait $$p || rc=1; done; exit $$rc

train-ai-route: $(addprefix $(AI_BUILD)/bin/,route_tool train_route export_route)  ## learned routing policy: teacher data + BC + one DAgger round (drivers/route/README.md)
	@mkdir -p $(AI_ROUTE_WORK)/stage
	$(AI_BUILD)/bin/route_tool gen-jobs 31 1000 30 0.5 $(AI_ROUTE_WORK)/jobs0.json
	$(AI_BUILD)/bin/route_tool gen-jobs 32 1000 30 0.5 $(AI_ROUTE_WORK)/jobs1.json
	ROUTE_TOOL_CAP=96 $(AI_BUILD)/bin/route_tool gen $(AI_ROUTE_WORK)/jobs0.json $(AI_ROUTE_WORK)/d0.bin - 1.0 401 heuristic 16
	$(AI_BUILD)/bin/train_route --data $(AI_ROUTE_WORK)/d0.bin --out $(AI_ROUTE_WORK)/run0 --hidden 64 --tau 0.5 --epochs 30 --seed 0 --budget-s $(AI_ROUTE_BC_BUDGET_S)
	$(AI_BUILD)/bin/export_route $(AI_ROUTE_WORK)/run0 --root $(AI_ROUTE_WORK)/stage --id round0 --default
	ROUTE_TOOL_CAP=96 $(AI_BUILD)/bin/route_tool gen $(AI_ROUTE_WORK)/jobs1.json $(AI_ROUTE_WORK)/d1.bin $(AI_ROUTE_WORK)/stage/round0 0.3 402 heuristic 16
	$(AI_BUILD)/bin/train_route --data $(AI_ROUTE_WORK)/d0.bin $(AI_ROUTE_WORK)/d1.bin --out $(AI_ROUTE_WORK)/run1 --hidden 64 --tau 0.5 --epochs 30 --seed 0 --budget-s $(AI_ROUTE_BC_BUDGET_S)
	$(AI_BUILD)/bin/export_route $(AI_ROUTE_WORK)/run1 --root $(MODELS)/route --id $(AI_ROUTE_ID) --default

train-ai-smoke smoke: $(addprefix $(AI_BUILD)/bin/,train_synth export_synth eval_synth route_tool train_route export_route models_verify)  ## ~5 min end-to-end into AI_BUILD/smoke (synth learns and beats greedy, router BC, export + verify); never writes MODELS
	./smoke.sh $(AI_BUILD)

train-ai-route-ppo: $(addprefix $(AI_BUILD)/bin/,train_route_ppo export_route)  ## optional router PPO fine-tune: make train-ai-route-ppo INIT=<router dir> BUDGET_S=<s>
	@test -n "$(INIT)" -a -n "$(BUDGET_S)" || { echo "usage: make train-ai-route-ppo INIT=<router dir> BUDGET_S=<seconds>"; exit 2; }
	$(AI_BUILD)/bin/train_route_ppo --init $(INIT) --out $(AI_ROUTE_WORK)/ppo_run --budget-s $(BUDGET_S)
