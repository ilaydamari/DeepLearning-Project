"""
Automated Experiment Runner for Assignment 3
==========================================
Runs all 30 required experiments (5 melodies × 3 seed words × 2 models)
for generating the complete results needed for 100% grade.
"""

import os
import subprocess
import json
import pandas as pd
from datetime import datetime

# Configuration
MIDI_FILES = [
    "1910_Fruitgum_Company_-_Simon_Says.mid",
    "2_Unlimited_-_Get_Ready_for_This.mid", 
    "2_Unlimited_-_Let_the_Beat_Control_Your_Body.mid",
    "2_Unlimited_-_Tribal_Dance.mid",
    "2_Unlimited_-_Twilight_Zone.mid"
]

SEED_WORDS = ["love", "night", "dream"]

MODELS = {
    "concatenation": "models/melody_concatenation_model.pth",
    "conditioning": "models/melody_conditioning_model.pth"
}

MIDI_DIR = "data/midi/test/"
OUTPUT_DIR = "experiment_results/"
MAX_LENGTH = 80
TEMPERATURE = 0.8

def create_results_table(output_dir):
    """
    Create organized table of all 30 generated songs for the report appendix.
    """
    
    results_data = []
    
    for model_name in MODELS.keys():
        for midi_file in MIDI_FILES:
            for seed_word in SEED_WORDS:
                # Read generated lyrics
                output_file = f"{output_dir}{model_name}_{midi_file[:-4]}_{seed_word}.txt"
                
                if os.path.exists(output_file):
                    with open(output_file, 'r', encoding='utf-8') as f:
                        lyrics = f.read().strip()
                        
                        # Extract first few lines for preview
                        lines = lyrics.split('\n')
                        preview = '\n'.join(lines[:4]) if len(lines) >= 4 else lyrics
                        
                        results_data.append({
                            'Model': model_name.title(),
                            'MIDI File': midi_file[:-4].replace('_', ' '),
                            'Seed Word': seed_word.title(),
                            'Generated Lyrics Preview': preview,
                            'Full Length': len(lyrics.split()),
                            'Line Count': len(lines)
                        })
                else:
                    results_data.append({
                        'Model': model_name.title(),
                        'MIDI File': midi_file[:-4].replace('_', ' '),
                        'Seed Word': seed_word.title(),
                        'Generated Lyrics Preview': '[GENERATION FAILED]',
                        'Full Length': 0,
                        'Line Count': 0
                    })
    
    # Create DataFrame and save as CSV and Excel
    df = pd.DataFrame(results_data)
    
    # Save as CSV for easy viewing
    csv_path = f"{output_dir}all_generated_songs_table.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8')
    
    # Save as Excel with formatting for the report
    excel_path = f"{output_dir}all_generated_songs_table.xlsx"
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Generated Songs', index=False)
        
        # Get the workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets['Generated Songs']
        
        # Format headers
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'bg_color': '#D7E4BC'
        })
        
        # Format lyrics preview column
        lyrics_format = workbook.add_format({
            'text_wrap': True,
            'valign': 'top',
            'font_size': 10
        })
        
        # Apply formatting
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
            
        # Set column widths
        worksheet.set_column('A:A', 15)  # Model
        worksheet.set_column('B:B', 25)  # MIDI File  
        worksheet.set_column('C:C', 12)  # Seed Word
        worksheet.set_column('D:D', 50)  # Lyrics Preview
        worksheet.set_column('E:E', 12)  # Length
        worksheet.set_column('F:F', 12)  # Line Count
        
        # Format lyrics column
        worksheet.set_column('D:D', 50, lyrics_format)
    
    print(f"\n📊 Results table created:")
    print(f"   CSV: {csv_path}")
    print(f"   Excel: {excel_path}")
    
    return df

def run_experiment(model_name, model_path, midi_file, seed_word):
    """Run a single generation experiment."""
    
    midi_path = os.path.join(MIDI_DIR, midi_file)
    output_file = f"{OUTPUT_DIR}{model_name}_{midi_file[:-4]}_{seed_word}.txt"
    
    cmd = [
        "python", "generate_melody.py",
        "--model_path", model_path,
        "--midi_file", midi_path,
        "--seed_words", seed_word,
        "--max_length", str(MAX_LENGTH),
        "--temperature", str(TEMPERATURE),
        "--output_file", output_file
    ]
    
    print(f"Running: {model_name} + {midi_file} + '{seed_word}'")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Success")
            return True
        else:
            print(f"❌ Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def main():
    """Run all experiments."""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Track results
    results = {
        "timestamp": datetime.now().isoformat(),
        "experiments": [],
        "summary": {"total": 0, "success": 0, "failed": 0}
    }
    
    experiment_count = 0
    success_count = 0
    
    print("🎵 Starting Full Experiment Suite for Assignment 3 🎵")
    print("=" * 60)
    print(f"Total experiments: {len(MIDI_FILES)} × {len(SEED_WORDS)} × {len(MODELS)} = {len(MIDI_FILES) * len(SEED_WORDS) * len(MODELS)}")
    print()
    
    # Run all combinations
    for model_name, model_path in MODELS.items():
        print(f"\n🤖 Model: {model_name.upper()}")
        print("-" * 40)
        
        for midi_file in MIDI_FILES:
            for seed_word in SEED_WORDS:
                experiment_count += 1
                
                print(f"\n[{experiment_count:2d}] ", end="")
                
                success = run_experiment(model_name, model_path, midi_file, seed_word)
                if success:
                    success_count += 1
                
                # Record result
                results["experiments"].append({
                    "experiment_id": experiment_count,
                    "model": model_name,
                    "midi_file": midi_file,
                    "seed_word": seed_word,
                    "success": success
                })
    
    # Save summary
    results["summary"]["total"] = experiment_count
    results["summary"]["success"] = success_count
    results["summary"]["failed"] = experiment_count - success_count
    
    with open(f"{OUTPUT_DIR}experiment_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Create organized results table for report appendix
    print("\n📋 Creating organized results table for report...")
    results_df = create_results_table(OUTPUT_DIR)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("🎉 EXPERIMENT SUITE COMPLETE 🎉")
    print("=" * 60)
    print(f"Total experiments: {experiment_count}")
    print(f"Successful: {success_count}")
    print(f"Failed: {experiment_count - success_count}")
    print(f"Success rate: {success_count/experiment_count*100:.1f}%")
    print()
    print(f"📁 Results saved in: {OUTPUT_DIR}")
    print(f"📊 Summary: {OUTPUT_DIR}experiment_summary.json")
    
    if success_count == experiment_count:
        print("\n✅ ALL EXPERIMENTS SUCCESSFUL - READY FOR GRADE 100!")
        print("\n📝 CRITICAL REPORT WRITING GUIDELINES:")
        print("=" * 50)
        print("\n🎯 FOCUS ON QUALITATIVE ANALYSIS (NOT JUST TECHNICAL PERFORMANCE):")
        print("\n1. 🎵 MELODY INFLUENCE ANALYSIS:")
        print("   • How do different MIDI files affect lyrical themes?")
        print("   • Does 'Simon Says' create different content than 'Tribal Dance'?")
        print("   • Are faster tempo songs generating more energetic words?")
        print("\n2. 🌱 SEED WORD IMPACT ANALYSIS:")
        print("   • How does starting with 'love' vs 'night' vs 'dream' change the song?")
        print("   • Do seed words maintain thematic consistency throughout?")
        print("   • Which seed word produces most coherent songs?")
        print("\n3. 🤖 MODEL COMPARISON (CONCATENATION VS CONDITIONING):")
        print("   • Which approach produces more melody-aligned lyrics?")
        print("   • Does Conditioning model show better thematic consistency?")
        print("   • Are there noticeable differences in creativity/diversity?")
        print("\n4. 🎼 SONG STRUCTURE QUALITY:")
        print("   • Do the generated lyrics follow verse/chorus patterns?")
        print("   • Are line lengths appropriate for singing?")
        print("   • Is there repetition that mimics real songs?")
        print("\n📊 APPENDIX REQUIREMENTS:")
        print("   ✅ Include the Excel table: all_generated_songs_table.xlsx")
        print("   ✅ Organize by: Model → MIDI File → Seed Words")
        print("   ✅ Show preview of each song + full statistics")
        print("\n🎨 REPORT STRUCTURE (6 pages max):")
        print("   1. Introduction (0.5 page)")
        print("   2. Architecture Description (1.5 pages)")
        print("   3. Training Results + TensorBoard graphs (1 page)")
        print("   4. ⭐ QUALITATIVE ANALYSIS (2.5 pages) - MOST IMPORTANT!")
        print("      → Melody influence patterns")
        print("      → Seed word impact")
        print("      → Model comparison examples")
        print("      → Song structure assessment")
        print("   5. Conclusions (0.5 page)")
        print("\n📎 FORMAT REQUIREMENTS:")
        print("   • DOCX + PDF versions")
        print("   • Calibri 12pt, 2.5cm margins")
        print("   • Appendix with full 30-song table")
        print("   • Reference TensorBoard graphs in appendix")
        
    else:
        print(f"\n⚠️ {experiment_count - success_count} experiments failed - check logs")

if __name__ == "__main__":
    main()