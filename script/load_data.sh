# download Kontext Bench
huggingface-cli download --repo-type dataset --resume-download black-forest-labs/kontext-bench --local-dir data/Kontext-Bench

# download GEdit Bench
huggingface-cli download --repo-type dataset --resume-download GEdit --local-dir data/GEdit-Bench

# preprocess datasets
python data/preprocess.py
