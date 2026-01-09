import os
import subprocess
import sys

# Parameters - change these as needed
# MODEL_NAME should be the name of the .nc file in PREDS_PATH (without .nc extension)
# Examples: 'vit_ASYM', 'vit_CRPS', 'vit_BerGamma'
params = {
    'MODEL_NAME': os.getenv('MODEL_NAME', 'vit_ASYM')
}

# Paths to the scripts
eval_dir = os.path.dirname(os.path.abspath(__file__))
scripts = [
    os.path.join(eval_dir, 'standard_metrics.py'),
    os.path.join(eval_dir, 'histogram_comparison.py'),
    os.path.join(eval_dir, 'psd_comparison.py'),
    os.path.join(eval_dir, 'daily_comparison.py'),
    os.path.join(eval_dir, 'ensemble_comparison.py')  # Will skip if no ensemble dimension
]

def merge_pdfs(pdf_list, output_path):
    """Merge multiple PDFs into one."""
    try:
        from pypdf import PdfWriter
    except ImportError:
        try:
            from PyPDF2 import PdfWriter
        except ImportError:
            print("\n[WARNING] Could not find 'pypdf' or 'PyPDF2' library.")
            print("Skipping PDF merge. Install with: pip install pypdf")
            return False

    writer = PdfWriter()
    merged_count = 0
    for pdf in pdf_list:
        if os.path.exists(pdf):
            writer.append(pdf)
            merged_count += 1
            print(f"  Added: {os.path.basename(pdf)}")
        else:
            print(f"  Skipped (not found): {os.path.basename(pdf)}")
    
    if merged_count > 0:
        with open(output_path, "wb") as f:
            writer.write(f)
        print(f"\nSuccessfully merged {merged_count} PDFs into: {output_path}")
        return True
    else:
        print("\nNo PDFs to merge.")
        return False

def main():
    # Update environment variables
    env = os.environ.copy()
    env.update(params)

    model_name = params['MODEL_NAME']
    figs_path = '/gpfs/projects/meteo/WORK/gonzabad/test-deep4downscaling/eval_ci/figs'
    
    # Ensure output directory exists
    os.makedirs(figs_path, exist_ok=True)
    
    output_pdfs = []
    
    print(f"\n{'='*60}")
    print(f"Running evaluation for model: {model_name}")
    print(f"{'='*60}\n")
    
    # 1. Run scripts
    for script in scripts:
        print(f"\n{'─'*60}")
        print(f"Running {os.path.basename(script)}")
        print(f"{'─'*60}")
        try:
            # Run the script as standalone process
            result = subprocess.run([sys.executable, script], env=env, 
                                   capture_output=False, text=True)
            
            # Construct the expected PDF path
            script_basename = os.path.basename(script).replace('.py', '')
            pdf_name = f"{model_name}_{script_basename}.pdf"
            pdf_path = os.path.join(figs_path, pdf_name)
            
            # Check if PDF was created
            if os.path.exists(pdf_path):
                output_pdfs.append(pdf_path)
                print(f"✓ Generated: {pdf_name}")
            else:
                print(f"⚠ No PDF generated (may be expected for ensemble_comparison if no ensemble)")
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Error running {script}: {e}")
            # Don't exit, continue with other scripts
        except Exception as e:
            print(f"✗ Unexpected error: {e}")

    # 2. Merge resulting PDFs
    if len(output_pdfs) > 0:
        print(f"\n{'='*60}")
        print("Merging PDFs...")
        print(f"{'='*60}\n")
        merged_output = os.path.join(figs_path, f"{model_name}_FULL_REPORT.pdf")
        if merge_pdfs(output_pdfs, merged_output):
            # 3. Remove the individual PDFs (optional)
            print("\nCleaning up individual PDFs...")
            for pdf in output_pdfs:
                try:
                    os.remove(pdf)
                    print(f"  Removed: {os.path.basename(pdf)}")
                except Exception as e:
                    print(f"  Could not remove {os.path.basename(pdf)}: {e}")
    else:
        print("\n⚠ No PDFs were generated.")
    
    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

