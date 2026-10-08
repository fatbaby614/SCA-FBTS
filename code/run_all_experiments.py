#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
One-click script to run all sleep staging experiments and generate all required figures.

This script orchestrates entire pipeline (19 main steps):
1. Runs main sleep evaluation on Sleep-EDF dataset (with t-SNE visualization)
2. Generates efficiency vs accuracy visualization
3. Generates algorithm comparison bars
4. Generates hypnogram comparison (example)
5. Runs noise robustness experiment (Sleep-EDF, multi-repeat with controlled SNR)
6. Runs channel ablation experiment (ISRUC)
7. Runs algorithm ablation experiment (Sleep-EDF)
8. Generates per-stage F1 analysis
9. Generates extended Table I
10. Runs ISRUC dataset experiments (40 subjects max)
11. Runs DREAMS patients dataset experiments (up to 40 subjects)
12. Generates statistical significance tests (SUBJECT-level Friedman + Wilcoxon)
13. Runs parameter sensitivity analysis
14. Generates training time comparison figure
15. Runs cross-dataset generalization experiment
16. Runs frequency band contribution analysis
17. Runs channel importance analysis
18. Runs single-channel evaluation (channel-matched comparison, Sleep-EDF)
19. Generates channel-matched comparison table/figure

Usage:
    python run_all_experiments.py              # Run all steps with full data
    python run_all_experiments.py --quick      # Run all steps with small samples (smoke test)
    python run_all_experiments.py --step 1     # Run only step 1
    python run_all_experiments.py --step 19    # Run only step 19
"""
import argparse
import os
import subprocess
import time
import sys
from pathlib import Path

# Algorithm name mapping for consistent labeling
ALGORITHM_ALIAS = {
    'SCA-FBTS': 'SCA-FBTS (Ours)',
    'SSC-SleepNet': 'SSC-SleepNet',
    'HandcraftedFeatures+RF': 'Handcrafted+RF',
    'DeepSleepNet': 'DeepSleepNet',
    'TinySleepNet': 'TinySleepNet',
    'RiemannTangentSpace': 'RiemannTangentSpace',
    'MDM': 'MDM'
}

CONTINUE_ON_ERROR = False
FAILED_COMMANDS = []

# Directory of this script (code/) and project root - cross-platform (Windows/Linux)
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent

class TeeOutput:
    """Dual output: console + file."""
    def __init__(self, filepath):
        self.filepath = filepath
        self.file = open(filepath, 'w', encoding='utf-8', buffering=1)
        self._original_stdout = sys.stdout
    
    def write(self, text):
        self._original_stdout.write(text)
        self.file.write(text)
        self.file.flush()
    
    def flush(self):
        self._original_stdout.flush()
        self.file.flush()
    
    def close(self):
        self._original_stdout.flush()
        self.file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()

def run_command(cmd, cwd=None, check=True, step_desc=None):
    """Run a command and return output.

    If check=True, raise RuntimeError on non-zero return code.
    step_desc: optional human-readable description of the step's purpose
               (printed to the log for easier debugging).
    """
    start_time = time.time()
    print(f"\n[{time.strftime('%H:%M:%S')}] >>> Running command:")
    print(f"    {cmd}")
    if step_desc:
        print(f"    Purpose: {step_desc}")
    print("-" * 80)
    
    # Run subprocesses from code/ so sibling modules and config package resolve correctly
    if cwd is None:
        cwd = CODE_DIR

    # Force child Python processes to be unbuffered so print() output flows
    # to us in real-time (otherwise Python uses block buffering on pipes,
    # hiding progress for hours during long computations). Two layers:
    #   1. PYTHONUNBUFFERED=1 env var (POSIX + Windows both respect this)
    #   2. "-u" prefix on "python" in the command string (Python's own flag
    #      that forces unbuffered mode regardless of buffering detection).
    import os as _os
    env = dict(_os.environ)
    env['PYTHONUNBUFFERED'] = '1'

    # Prepend "-u" to python commands so Python itself runs unbuffered.
    # Use sys.executable instead of bare "python" so child processes run in the
    # SAME interpreter/environment that launched this script (avoids silently
    # picking a different Python from PATH with mismatched packages).
    cmd_unbuffered = cmd
    if cmd.startswith('python '):
        cmd_unbuffered = f'"{sys.executable}" -u ' + cmd[len('python '):]

    # Use Popen to capture stdout/stderr in real-time and pipe it to TeeOutput
    process = subprocess.Popen(
        cmd_unbuffered, shell=True, cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    
    # Read output line by line as it is generated
    for line in process.stdout:
        print(line, end='')
        
    process.wait()
    elapsed = time.time() - start_time
    print(f"\n[{time.strftime('%H:%M:%S')}] >>> Command finished in {elapsed:.1f}s "
          f"(exit code {process.returncode})")
    
    # Create a mock result object to maintain API compatibility
    class MockResult:
        def __init__(self, returncode):
            self.returncode = returncode
            
    result = MockResult(process.returncode)
    
    if check and result.returncode != 0:
        if CONTINUE_ON_ERROR:
            print(f"\n[!] WARNING: Command failed with exit code {result.returncode} but continuing...")
            print(f"    Failed step: {step_desc or 'N/A'}")
            FAILED_COMMANDS.append(cmd[:150] + '...' if len(cmd) > 150 else cmd)
        else:
            # Print detailed error context before raising
            import traceback
            print("\n" + "!" * 80)
            print(f"[!] ERROR: Command failed with exit code {result.returncode}")
            print(f"    Step: {step_desc or 'N/A'}")
            print(f"    Elapsed: {elapsed:.1f}s")
            print(f"    Command: {cmd}")
            print("!" * 80)
            traceback.print_stack()
            raise RuntimeError(f"Command failed ({result.returncode}): {cmd}")
    return result

def main():
    parser = argparse.ArgumentParser(description="Run all sleep staging experiments and generate figures")
    parser.add_argument('--output-dir', default=str(PROJECT_ROOT / 'results'), help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Quick test mode for smoke testing')
    parser.add_argument('--step', type=int, default=None, help='Run specific step (1-20). If not specified, run all steps')
    parser.add_argument('--from-step', type=int, default=None,
                        help='Run all steps from this step number onwards (inclusive). '
                             'Use to resume after already-completed earlier steps, e.g. --from-step 5.')
    parser.add_argument('--continue-on-error', action='store_true', help='Continue running subsequent steps if a step fails')
    parser.add_argument('--sleep-edf-path', type=str, default=None, help='Path to Sleep-EDF dataset')
    parser.add_argument('--isruc-path', type=str, default=None, help='Path to ISRUC dataset')
    parser.add_argument('--dreams-path', type=str, default=None, help='Path to DREAMS patients dataset')
    args = parser.parse_args()
    
    global CONTINUE_ON_ERROR
    CONTINUE_ON_ERROR = args.continue_on_error
    
    # Detect operating system and set default paths
    if os.name == 'nt':
        # Windows
        default_sleep_edf = "E:/datasets/Sleep/sleep-edf-database-expanded-1.0.0"
        default_isruc = "E:/datasets/Sleep/ISRUC-Sleep"
        default_dreams = "E:/datasets/Sleep/DREAMS/DatabasePatients"
    else:
        # Linux/macOS
        default_sleep_edf = "/mnt/data1/home/tanhuang/datasets/sleep-edf-database-expanded-1.0.0"
        default_isruc = "/mnt/data1/home/tanhuang/datasets/ISRUC-SLEEP"
        default_dreams = "/mnt/data1/home/tanhuang/datasets/DREAMS/DatabasePatients"

    # Override with command-line arguments if provided
    SLEEP_EDF_PATH = args.sleep_edf_path if args.sleep_edf_path else default_sleep_edf
    ISRUC_PATH = args.isruc_path if args.isruc_path else default_isruc
    DREAMS_PATIENTS_PATH = args.dreams_path if args.dreams_path else default_dreams
    
    # Pre-validate Dataset Paths
    print("\nValidating dataset paths...")

    def should_run(n):
        """Decide whether step n runs, honouring --step and --from-step."""
        if args.step is not None:
            return args.step == n
        if args.from_step is not None:
            return n >= args.from_step
        return True

    if args.from_step is not None:
        print(f"[RESUME] Running steps {args.from_step} onwards (--from-step {args.from_step})")

    missing_paths = []
    if not Path(SLEEP_EDF_PATH).exists():
        missing_paths.append(f"Sleep-EDF ({SLEEP_EDF_PATH})")
    if not Path(ISRUC_PATH).exists():
        missing_paths.append(f"ISRUC ({ISRUC_PATH})")
    if not Path(DREAMS_PATIENTS_PATH).exists():
        missing_paths.append(f"DREAMS ({DREAMS_PATIENTS_PATH})")
        
    if missing_paths:
        print("\n[!] CRITICAL ERROR: The following dataset paths do not exist:")
        for mp in missing_paths:
            print(f"  - {mp}")
        print("\nPlease update the paths in the script or provide them via command line arguments.")
        print("Example: python run_all_experiments.py --sleep-edf-path \"C:/data/sleep_edf\"")
        print("\nAborting execution to prevent failed experiments.")
        sys.exit(1)

    # Environment info for debugging (written to log file as well)
    import platform
    print("\n" + "=" * 80)
    print("[ENV] Environment details:")
    print(f"  Platform: {platform.platform()} ({os.name})")
    print(f"  Python: {sys.version.split()[0]}")
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        print(f"  PyTorch: not importable ({exc})")
    try:
        import mne
        print(f"  MNE: {mne.__version__}")
    except Exception:
        pass
    print("=" * 80)
    
    global_start_time = time.time()
    
    print("=" * 80)
    print("Running All Sleep Staging Experiments")
    print("=" * 80)
    print(f"Operating System: {os.name}")
    print(f"Sleep-EDF Path: {SLEEP_EDF_PATH}")
    print(f"ISRUC Path: {ISRUC_PATH}")
    print(f"DREAMS Patients Path: {DREAMS_PATIENTS_PATH}")
    print("=" * 80)
    
    # Create output directories if they don't exist
    output_dir = Path(args.output_dir)
    sleep_edf_output = output_dir / "sleep_edf"
    isruc_output = output_dir / "isruc"
    dreams_output = output_dir / "dreams"
    common_output = output_dir / "common"
    
    for dir_path in [output_dir, sleep_edf_output, isruc_output, dreams_output, common_output]:
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            # Test write permissions
            test_file = dir_path / ".write_test"
            test_file.touch()
            test_file.unlink()
        except PermissionError:
            print(f"\n[!] CRITICAL ERROR: No write permission for output directory: {dir_path}")
            print("Please run the script with elevated permissions or choose a different output directory (--output-dir).")
            sys.exit(1)
        except Exception as e:
            print(f"\n[!] CRITICAL ERROR: Could not create or write to output directory: {dir_path}\nReason: {e}")
            sys.exit(1)
            
    # Initialize log file
    from datetime import datetime
    log_filename = f"experiment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_path = output_dir / log_filename
    print(f"\nLog file: {log_path}")
    print("=" * 80)
    
    # Enable dual output (console + file)
    tee = TeeOutput(log_path)
    old_stdout = sys.stdout
    sys.stdout = tee
    print("Running All Sleep Staging Experiments")
    print("=" * 80)
    print(f"Operating System: {os.name}")
    print(f"Sleep-EDF Path: {SLEEP_EDF_PATH}")
    print(f"ISRUC Path: {ISRUC_PATH}")
    print(f"DREAMS Patients Path: {DREAMS_PATIENTS_PATH}")
    print("=" * 80)
    
    # Step1: Run main sleep evaluation with t-SNE for Sleep-EDF
    print("\n" + "=" * 80)
    print("Step 1: Running main sleep evaluation for Sleep-EDF (with t-SNE)")
    print("=" * 80)
    
    if should_run(1):
        # In quick mode, use fewer subjects for faster testing
        if args.quick:
            eval_cmd = f"python evaluate_sleep.py "
            eval_cmd += f"--dataset sleep_edf "
            eval_cmd += f"--data-path \"{SLEEP_EDF_PATH}\" "
            eval_cmd += f"--output-dir \"{sleep_edf_output}\" "
            eval_cmd += "--subjects 0~9 "
            eval_cmd += "--algorithms SCA-FBTS "
            eval_cmd += "--cv-mode stratified "
            eval_cmd += "--summary-view compact "
            eval_cmd += "--tsne "
            eval_cmd += "--tsne-max-samples 200"
        else:
            eval_cmd = f"python evaluate_sleep.py "
            eval_cmd += f"--dataset sleep_edf "
            eval_cmd += f"--data-path \"{SLEEP_EDF_PATH}\" "
            eval_cmd += f"--output-dir \"{sleep_edf_output}\" "
            eval_cmd += "--subjects 0~77 "  # All 78 Sleep-Cassette subjects (Expanded v1.0.0)
            eval_cmd += "--algorithms SCA-FBTS SSC-SleepNet HandcraftedFeatures+RF DeepSleepNet TinySleepNet RiemannTangentSpace MDM "
            eval_cmd += "--cv-mode subject "  # Use subject-wise CV for consistency
            eval_cmd += "--summary-view compact "
            eval_cmd += "--tsne "
            eval_cmd += "--tsne-max-samples 500"
        
        result = run_command(eval_cmd, step_desc="Main sleep evaluation on Sleep-EDF "
                              "(subject-wise 5-fold CV, 7 algorithms, t-SNE)")
    
    # Find the latest sleep evaluation summary file (Only strictly needed if running steps 2,3,8,9)
    summary_files = list(sleep_edf_output.glob('sleep_evaluation_summary_*.csv'))
    summary_file = None
    if summary_files:
        summary_file = sorted(summary_files, key=lambda x: x.stat().st_mtime)[-1]
        print(f"\nLatest summary file: {summary_file}")
    elif args.step in [None, 2, 3, 9]:
        print("Error: No sleep evaluation summary file found required for this step. Please run Step 1 first.")
        if CONTINUE_ON_ERROR:
            FAILED_COMMANDS.append("Find summary file")
        else:
            sys.exit(1)
            
    # Find the latest per-stage F1 file
    per_stage_files = list(sleep_edf_output.glob('per_stage_f1_*.csv'))
    per_stage_file = None
    if per_stage_files:
        per_stage_file = sorted(per_stage_files, key=lambda x: x.stat().st_mtime)[-1]
        print(f"Latest per-stage F1 file: {per_stage_file}")
    elif args.step in [None, 8]:
        print("Error: No per_stage_f1_*.csv file found. Please run Step 1 first.")
        if not CONTINUE_ON_ERROR:
            sys.exit(1)
    
    # Step 2: Generate efficiency vs accuracy visualization
    print("\n" + "=" * 80)
    print("Step 2: Generating efficiency vs accuracy visualization")
    print("=" * 80)
    
    if should_run(2):
        if summary_file is None:
            print("[SKIP] Step 2: no sleep_evaluation_summary_*.csv from Step 1; skipping.")
        else:
            efficiency_cmd = f"python generate_efficiency_vs_accuracy.py "
            efficiency_cmd += f"--csv {summary_file} "
            efficiency_cmd += f"--output \"{common_output}\""

            run_command(efficiency_cmd, step_desc="Efficiency vs accuracy visualization from summary CSV")
    
    # Step 3: Generate algorithm comparison bars
    print("\n" + "=" * 80)
    print("Step 3: Generating algorithm comparison bars")
    print("=" * 80)

    if should_run(3):
        if summary_file is None:
            print("[SKIP] Step 3: no sleep_evaluation_summary_*.csv from Step 1; skipping.")
        else:
            import_text = f"import sys; sys.path.insert(0, r'{CODE_DIR.as_posix()}'); "
            import_text += f"import pandas as pd; "
            import_text += f"from visualization import plot_algorithm_comparison_bars; "
            import_text += f"df = pd.read_csv(r'{summary_file.as_posix()}'); "
            import_text += f"plot_algorithm_comparison_bars(df, save_path=r'{common_output.as_posix()}/algorithm_comparison_bars.png')"
            bars_cmd = f"python -c \"{import_text}\""

            run_command(bars_cmd, step_desc="Algorithm comparison bar chart from summary CSV")
    
    # Step 4: Generate example hypnogram comparison
    print("\n" + "=" * 80)
    print("Step 4: Generating example hypnogram comparison")
    print("=" * 80)
    
    if should_run(4):
        hypnogram_cmd = f"python generate_hypnogram.py "
        hypnogram_cmd += f"--dataset sleep_edf "
        hypnogram_cmd += f"--data-path \"{SLEEP_EDF_PATH}\" "
        hypnogram_cmd += f"--subject 1 "
        hypnogram_cmd += f"--algorithm SCA-FBTS "
        hypnogram_cmd += f"--output-dir \"{common_output}\" "
        
        if args.quick:
            hypnogram_cmd += "--compare-algorithms TinySleepNet"
        else:
            hypnogram_cmd += "--compare-algorithms TinySleepNet DeepSleepNet SSC-SleepNet"
            
        run_command(hypnogram_cmd, step_desc="Hypnogram comparison on an example subject")
    
    # Step 5: Run noise robustness experiment
    print("\n" + "=" * 80)
    print("Step 5: Running noise robustness experiment (Sleep-EDF)")
    print("=" * 80)
    
    if should_run(5):
        noise_cmd = f"python run_noise_robustness_experiment.py "
        noise_cmd += f"--dataset sleep_edf "
        noise_cmd += f"--data-path \"{SLEEP_EDF_PATH}\" "
        noise_cmd += f"--output-dir \"{common_output}\" "
        
        if args.quick:
            # Quick mode: only 1 algorithm + 1 noise realization (default, but explicit)
            noise_cmd += "--algorithms SCA-FBTS "
            noise_cmd += "--n-repeats 1 "
        else:
            # Full mode: all algorithms incl. deep baselines (reviewer R3-6)
            noise_cmd += "--algorithms SCA-FBTS SSC-SleepNet HandcraftedFeatures+RF "
            noise_cmd += "TinySleepNet DeepSleepNet "
            # Multiple independent noise realizations to report SD/CI (reviewer R3-6)
            noise_cmd += "--n-repeats 3 "
        
        run_command(noise_cmd, step_desc="Noise robustness experiment on Sleep-EDF "
                      "(controlled SNR, multi-repeat, incl. deep baselines)")
    
    # Step 6: Run channel ablation experiment
    print("\n" + "=" * 80)
    print("Step 6: Running channel ablation experiment (ISRUC)")
    print("=" * 80)
    
    if should_run(6):
        channel_ablation_cmd = f"python run_sleep_channel_ablation.py "
        channel_ablation_cmd += f"--dataset isruc "
        channel_ablation_cmd += f"--data-path \"{ISRUC_PATH}\" "
        channel_ablation_cmd += f"--output-dir \"{common_output}\" "
        
        if args.quick:
            # Quick mode: use only 2 subjects and 1 algorithm
            channel_ablation_cmd += "--subjects 1~2 "
            channel_ablation_cmd += "--algorithms SCA-FBTS "
        else:
            # Full mode: channel ablation only for SCA-FBTS
            # Deep learning models like SSC-SleepNet don't benefit from channel ablation analysis
            channel_ablation_cmd += "--subjects 1~40 "
            channel_ablation_cmd += "--algorithms SCA-FBTS "
        
        run_command(channel_ablation_cmd, step_desc="Channel ablation experiment on ISRUC "
                      "(1-6 channels)")
    
    # Step 7: Run algorithm ablation experiment
    print("\n" + "=" * 80)
    print("Step 7: Running algorithm ablation experiment")
    print("=" * 80)
    
    if should_run(7):
        algo_ablation_cmd = f"python run_fbts_ablation.py "
        algo_ablation_cmd += f"--dataset sleep_edf "
        algo_ablation_cmd += f"--data-path \"{SLEEP_EDF_PATH}\" "
        algo_ablation_cmd += f"--output-dir \"{common_output}\" "
        # Default cv-mode is 'subject' for consistency with main evaluation
        
        if args.quick:
            # Quick mode: use only 2 subjects for faster testing
            algo_ablation_cmd += " --subjects 0~1 --n-splits 2"  # Reduce n_splits for quick mode
        
        run_command(algo_ablation_cmd, step_desc="Algorithm ablation (SCA-FBTS without bands/smoothing)")
    
    # Step 8: Generate per-stage F1 analysis
    print("\n" + "=" * 80)
    print("Step 8: Generating per-stage F1 analysis")
    print("=" * 80)
    
    if should_run(8):
        if per_stage_file:
            per_stage_cmd = f"python visualization.py replot-per-stage-f1 --csv \"{per_stage_file.as_posix()}\" --output \"{common_output.as_posix()}/per_stage_f1.png\""
            run_command(per_stage_cmd, step_desc="Per-stage F1 analysis plot")
        else:
            print("Skipping Step 8 due to missing per_stage F1 file.")
    
    # Step 9: Generate extended Table I
    print("\n" + "=" * 80)
    print("Step 9: Generating extended Table I")
    print("=" * 80)
    
    if should_run(9):
        table_cmd = f"python generate_extended_table_i.py "
        table_cmd += f"--csv {summary_file} "
        table_cmd += f"--output-dir {common_output}"
        
        run_command(table_cmd, step_desc="Extended Table I generation (LaTeX/CSV)")
    
    # Step 10: Run ISRUC dataset experiments
    print("\n" + "=" * 80)
    print("Step 10: Running ISRUC dataset experiments")
    print("=" * 80)
    
    if should_run(10):
        isruc_cmd = f"python evaluate_sleep.py "
        isruc_cmd += f"--dataset isruc "
        isruc_cmd += f"--data-path \"{ISRUC_PATH}\" "
        isruc_cmd += f"--output-dir \"{isruc_output}\" "
        
        if args.quick:
            # Quick mode: use only 2 subjects and 2 algorithms for faster testing
            isruc_cmd += "--subjects 1~2 "
            isruc_cmd += "--algorithms SCA-FBTS SSC-SleepNet "
        else:
            isruc_cmd += "--subjects 1~40 "
            isruc_cmd += "--algorithms SCA-FBTS SSC-SleepNet HandcraftedFeatures+RF DeepSleepNet TinySleepNet RiemannTangentSpace MDM "
        
        isruc_cmd += "--cv-mode subject "  # Use subject-wise CV for consistency
        isruc_cmd += "--summary-view compact"
        
        run_command(isruc_cmd, step_desc="ISRUC dataset main evaluation "
                      "(within-dataset, channel-consistent)")

    # Step 11: Run DREAMS patients dataset experiments
    print("\n" + "=" * 80)
    print("Step 11: Running DREAMS patients dataset experiments")
    print("=" * 80)

    if should_run(11):
        dreams_cmd = f"python evaluate_sleep.py "
        dreams_cmd += f"--dataset dreams "
        dreams_cmd += f"--data-path \"{DREAMS_PATIENTS_PATH}\" "
        dreams_cmd += f"--dreams-database patients "
        dreams_cmd += f"--output-dir \"{dreams_output}\" "

        if args.quick:
            dreams_cmd += "--subjects 0~4 "
            dreams_cmd += "--algorithms SCA-FBTS "
        else:
            dreams_cmd += "--subjects 0~26 "  # DREAMS Patients dataset has 27 subjects
            dreams_cmd += "--algorithms SCA-FBTS SSC-SleepNet HandcraftedFeatures+RF DeepSleepNet TinySleepNet RiemannTangentSpace MDM "

        dreams_cmd += "--cv-mode subject "  # Use subject-wise CV for consistency
        dreams_cmd += "--summary-view compact"

        run_command(dreams_cmd, step_desc="DREAMS patients dataset main evaluation "
                      "(within-dataset)")

    # Step 12: Run Statistical Significance Tests (subject-level, reviewer R3-3)
    print("\n" + "=" * 80)
    print("Step 12: Generating statistical significance tests (SUBJECT-level)")
    print("=" * 80)
    
    if should_run(12):
        sig_cmd = f"python generate_significance_tests.py --results-dir {sleep_edf_output} --output-dir {common_output}/generated_tables"
        run_command(sig_cmd, step_desc="Subject-level statistical significance tests "
                      "(Friedman + Wilcoxon on per-subject metrics)")

    # Step 13: Run Parameter Sensitivity Analysis 
    print("\n" + "=" * 80)
    print("Step 13: Running parameter sensitivity analysis")
    print("=" * 80)
    
    if should_run(13):
        sens_cmd = f"python run_parameter_sensitivity.py "
        sens_cmd += f"--dataset sleep_edf "
        sens_cmd += f"--data-path \"{SLEEP_EDF_PATH}\" "
        sens_cmd += f"--output-dir {common_output}"
        
        if args.quick:
            sens_cmd += " --n-folds 2"
        else:
            sens_cmd += " --n-folds 5"
            
        run_command(sens_cmd, step_desc="Parameter sensitivity analysis")

    # Step 14: Generate Training Time Figure
    print("\n" + "=" * 80)
    print("Step 14: Generating training time comparison figure")
    print("=" * 80)
    
    if should_run(14):
        if summary_file:
            time_cmd = f"python generate_training_time_figure.py --csv {summary_file.as_posix()} --output {common_output.as_posix()}"
            run_command(time_cmd, step_desc="Training time comparison figure")
        else:
            print("Skipping Step 14 due to missing summary file.")
    
    # Step 15: Run Cross-Dataset Generalization Experiment
    print("\n" + "=" * 80)
    print("Step 15: Running cross-dataset generalization experiment")
    print("=" * 80)
    
    if should_run(15):
        cross_cmd = f"python run_cross_dataset.py "
        cross_cmd += f"--source-data-path \"{SLEEP_EDF_PATH}\" "
        cross_cmd += f"--target-data-path \"{ISRUC_PATH}\" "
        cross_cmd += f"--output-dir \"{common_output}\" "
        
        if args.quick:
            cross_cmd += "--quick"
            
        run_command(cross_cmd, step_desc="Cross-dataset generalization (train Sleep-EDF -> "
                      "test ISRUC, channel-matched 2ch)")
    
    # Step 16: Run Frequency Band Contribution Analysis
    print("\n" + "=" * 80)
    print("Step 16: Running frequency band contribution analysis")
    print("=" * 80)
    
    if should_run(16):
        band_cmd = f"python run_band_contribution_analysis.py "
        band_cmd += f"--dataset sleep_edf "
        band_cmd += f"--data-path \"{SLEEP_EDF_PATH}\" "
        band_cmd += f"--output-dir \"{common_output}/band_analysis\" "
        
        if args.quick:
            band_cmd += "--subjects 0~4 --n-folds 3"
        else:
            band_cmd += "--subjects 0~77 --n-folds 5"  # All 78 Sleep-Cassette subjects
            
        run_command(band_cmd, step_desc="Frequency band contribution analysis (8 SCA-FBTS bands)")
    
    # Step 17: Run Channel Importance Analysis
    print("\n" + "=" * 80)
    print("Step 17: Running channel importance analysis")
    print("=" * 80)
    
    if should_run(17):
        channel_cmd = f"python run_channel_importance_analysis.py "
        channel_cmd += f"--dataset sleep_edf "
        channel_cmd += f"--data-path \"{SLEEP_EDF_PATH}\" "
        channel_cmd += f"--output-dir \"{common_output}/channel_analysis\" "
        
        if args.quick:
            channel_cmd += "--subjects 0~4 --n-folds 3"
        else:
            channel_cmd += "--subjects 0~77 --n-folds 5"  # All 78 Sleep-Cassette subjects
            
        run_command(channel_cmd, step_desc="Channel importance analysis (ISRUC)")
    
    # Step 20: Causal vs non-causal temporal smoothing ablation (reviewer R3-8)
    print("\n" + "=" * 80)
    print("Step 20: Running causal vs non-causal temporal smoothing ablation")
    print("=" * 80)
    
    if should_run(20):
        causal_cmd = f"python run_causal_smoothing_ablation.py "
        causal_cmd += f"--dataset sleep_edf "
        causal_cmd += f"--data-path \"{SLEEP_EDF_PATH}\" "
        causal_cmd += f"--output-dir \"{common_output}\" "
        
        if args.quick:
            causal_cmd += "--subjects 0~9 --n-folds 3"
        else:
            causal_cmd += "--subjects 0~19 --n-folds 5"
            
        run_command(causal_cmd, step_desc="Causal vs non-causal temporal smoothing ablation "
                      "(R3-8 portability check)")
    
    # Print summary based on step parameter
    print("\n" + "=" * 80)
    if args.step is None:
        print("All experiments completed successfully!")
    else:
        print(f"Step {args.step} completed successfully!")
    print("=" * 80)
    print("Generated files:")
    print(f"- Sleep-EDF evaluation summary: {summary_file}")
    print(f"- ISRUC evaluation summary: {isruc_output}/sleep_evaluation_summary_*.csv")
    print(f"- DREAMS evaluation summary: {dreams_output}/sleep_evaluation_summary_*.csv")
    print(f"- Efficiency vs accuracy: {common_output}/algorithm_efficiency_vs_accuracy.png")
    print(f"- Algorithm comparison bars: {common_output}/algorithm_comparison_bars.png")
    print(f"- Hypnogram comparison: {common_output}/hypnogram_comparison.png")
    print(f"- Noise robustness curves: {common_output}/noise_robustness_curve_*.png")
    print(f"- Noise robustness bars: {common_output}/noise_robustness_bars_*.png")
    print(f"- Channel ablation curve: {common_output}/sleep_channel_ablation_curve_*.png")
    print(f"- Algorithm ablation results: {common_output}/fbts_ablation_results_*.json")
    print(f"- Per-stage F1 analysis: {common_output}/per_stage_f1.png")
    print(f"- Extended Table I: {common_output}/extended_table_i.tex")
    print(f"- Statistical Significance (subject-level): {common_output}/generated_tables/table_stat_friedman_subject_level.csv")
    print(f"- Channel-Matched Comparison: {common_output}/channel_matched_comparison.csv")
    print(f"- Parameter Sensitivity: {common_output}/parameter_sensitivity*.png")
    print(f"- Training Time Comparison: figures/training_time_comparison.png")
    print(f"- Cross-Dataset Generalization: {common_output}/cross_dataset_accuracy.png")
    print(f"- Frequency Band Contribution: {common_output}/band_analysis/band_contribution_results_*.csv")
    print(f"- Frequency Band Contribution Table: {common_output}/band_analysis/band_contribution_table_*.tex")
    print(f"- Channel Importance: {common_output}/channel_analysis/channel_importance_results_*.csv")
    print(f"- Channel Importance Table: {common_output}/channel_analysis/channel_importance_table_*.tex")
    print(f"- t-SNE visualizations: {sleep_edf_output}/sleep_tsne_*.png")
    print(f"- t-SNE trajectory: {sleep_edf_output}/sleep_tsne_trajectory_*.png")
    print(f"- t-SNE stability: {sleep_edf_output}/sleep_tsne_stability_*.png")
    print(f"- Combined t-SNE figure: {sleep_edf_output}/sleep_tsne_figure4_*.png")
    print(f"- Experiment log: {log_path}")
    print("=" * 80)
    
    if FAILED_COMMANDS:
        print("\n" + "!" * 80)
        print(f"WARNING: {len(FAILED_COMMANDS)} command(s) failed during execution:")
        for failed_cmd in FAILED_COMMANDS:
            print(f"  - {failed_cmd}")
        print("!" * 80)

    total_elapsed = time.time() - global_start_time
    print(f"\n[SUMMARY] Total pipeline time: {total_elapsed/60:.1f} minutes "
          f"({total_elapsed:.1f}s)")
    print(f"[SUMMARY] Steps succeeded: 20 - {len(FAILED_COMMANDS)} | Steps failed: {len(FAILED_COMMANDS)}")
    
    # Restore stdout and close log file
    sys.stdout = old_stdout
    tee.close()
    print(f"\nExperiment completed! Log saved to: {log_path}")

if __name__ == "__main__":
    main()