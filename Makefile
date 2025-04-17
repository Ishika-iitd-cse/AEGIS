CONFIG_DIR      = configs
RESULTS_DIR     = results
NUM_CONFIGS     = 50
ALGORITHMS      = ipf_with_1d ipf_without_1d sampling
NUM_MODELS      = 4

.PHONY: all generate run clean clean_results test model final

all: generate run

generate:
	@mkdir -p $(CONFIG_DIR)
	@echo "Generating configs for dataset $(DATASET_TAG) with algorithm $(CURRENT_ALGORITHM) and model1"
	@python3 generate_configs.py \
		--original data/$(DATASET_TAG)/original_data.csv \
		--target_variable "$(TARGET_VARIABLE)" \
		--model_script data/$(DATASET_TAG)/model1.py \
		--output_dir $(CONFIG_DIR) \
		--dataset_tag $(DATASET_TAG) \
		--num_configs $(NUM_CONFIGS) \
		--algorithm $(CURRENT_ALGORITHM)

run:
	@mkdir -p $(RESULTS_DIR)
	@PYTHONPATH=. python3 main.py --configs_dir $(CONFIG_DIR) --results_dir $(RESULTS_DIR)

clean_results:
	@echo "Cleaning results for dataset $(DATASET_TAG)"
	@rm -rf $(RESULTS_DIR)/*

clean:
	@echo "Full clean for dataset $(DATASET_TAG)"
	@rm -rf $(CONFIG_DIR)/*
	@rm -rf $(RESULTS_DIR)/*
	@rm -rf data/1/config*.csv data/2/config*.csv data/3/config*.csv
	@rm -rf metrics/__pycache__ __pycache__/ src/__pycache__

test:
	@mkdir -p test/$(DATASET_TAG)
	@$(MAKE) generate
	@for alg in $(ALGORITHMS); do \
		echo " Testing algorithm $$alg"; \
		sed -i -E 's/^(  algorithm:).*/\1 '"$$alg"'/' $(CONFIG_DIR)/*.yaml; \
		$(MAKE) clean_results; \
		$(MAKE) run; \
		mkdir -p test/$(DATASET_TAG)/$$alg; \
		cp $(RESULTS_DIR)/summary_results.csv test/$(DATASET_TAG)/$$alg/; \
	done

model:
	@mkdir -p test/$(DATASET_TAG)
	@for model_num in $(shell seq 1 $(NUM_MODELS)); do \
		echo " Switching to model$$model_num"; \
		if [ $$model_num -eq 1 ]; then \
			$(MAKE) clean; \
			$(MAKE) generate CURRENT_ALGORITHM=ipf_with_1d; \
		else \
			sed -i -E \
			  's|^(  model_script:).*|\1 data/$(DATASET_TAG)/model'"$$model_num"'.py|' \
			  $(CONFIG_DIR)/*.yaml; \
		fi; \
		$(MAKE) clean_results; \
		for alg in $(ALGORITHMS); do \
			echo "  Testing $$alg with model$$model_num"; \
			sed -i -E 's/^(  algorithm:).*/\1 '"$$alg"'/' $(CONFIG_DIR)/*.yaml; \
			$(MAKE) run; \
			mkdir -p test/$(DATASET_TAG)/model$$model_num/$$alg; \
			cp $(RESULTS_DIR)/summary_results.csv test/$(DATASET_TAG)/model$$model_num/$$alg/; \
		done; \
	done

final:
	@echo "Running full sweep for datasets 1,2,3"
	@rm -rf test
	@for dt in 1 2 3; do \
		if [ $$dt -eq 1 ]; then TV="Air Quality"; \
		elif [ $$dt -eq 2 ]; then TV="is_booking"; \
		else TV="income"; fi; \
		echo "=== DATASET $$dt (target=$$TV) ==="; \
		$(MAKE) clean DATASET_TAG=$$dt TARGET_VARIABLE="$$TV" CURRENT_ALGORITHM=ipf_with_1d; \
		mkdir -p test/configs/$$dt; \
		$(MAKE) model DATASET_TAG=$$dt TARGET_VARIABLE="$$TV"; \
		cp $(CONFIG_DIR)/*.yaml test/configs/$$dt/; \
	done
