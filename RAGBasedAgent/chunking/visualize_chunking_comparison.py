import os
import json
import matplotlib.pyplot as plt
import pandas as pd

def visualize_chunking_comparison(json_path):
    """Visualize chunking comparison results from JSON file"""
    # Load JSON results
    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    if not results.get('comparison_table'):
        print("No comparison data available.")
        return
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(results['comparison_table'])
    
    # Create output directory
    output_dir = "chunking_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create base filename
    base_filename = os.path.splitext(os.path.basename(json_path))[0]
    
    # 1. Create bar chart for overall scores
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['name'], df['overall_score'], color='skyblue')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.3f}', ha='center', va='bottom')
    
    plt.title('Overall RAGAS Score by Chunking Strategy')
    plt.xlabel('Chunking Strategy')
    plt.ylabel('Overall Score')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{base_filename}_overall_scores.png")
    
    # 2. Create radar chart for metrics comparison
    metrics = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
    
    # Create radar chart
    plt.figure(figsize=(10, 8))
    
    # Number of metrics
    N = len(metrics)
    
    # What will be the angle of each axis in the plot (divide the plot / number of metrics)
    angles = [n / float(N) * 2 * 3.14159 for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create subplot with polar projection
    ax = plt.subplot(111, polar=True)
    
    # Draw one axis per metric and add labels
    plt.xticks(angles[:-1], metrics)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=8)
    plt.ylim(0, 1)
    
    # Plot each strategy
    for i, row in df.iterrows():
        values = [row['faithfulness'], row['answer_relevancy'], 
                 row['context_precision'], row['context_recall']]
        values += values[:1]  # Close the loop
        
        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=row['name'])
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('RAGAS Metrics by Chunking Strategy')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{base_filename}_metrics_radar.png")
    
    # 3. Create processing time vs chunks count scatter plot
    plt.figure(figsize=(10, 6))
    
    # Create scatter plot
    for i, row in df.iterrows():
        plt.scatter(row['chunk_count'], row['processing_time'], 
                   s=100, label=row['name'])
        
        # Add labels
        plt.annotate(row['name'], 
                    (row['chunk_count'], row['processing_time']),
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    
    plt.title('Processing Time vs. Chunk Count')
    plt.xlabel('Number of Chunks')
    plt.ylabel('Processing Time (seconds)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{base_filename}_time_vs_chunks.png")
    
    print(f"Visualizations saved to {output_dir}/ directory")
    
    # Generate a simple HTML report with the visualizations
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chunking Strategy Comparison - PR #{results['pr_number']}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .chart {{ margin-bottom: 40px; }}
            .best {{ background-color: #d4edda; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Chunking Strategy Comparison</h1>
            <h2>PR #{results['pr_number']} - {results.get('timestamp', '')}</h2>
            
            <h3>Comparison Table</h3>
            <table>
                <tr>
                    <th>Strategy</th>
                    <th>Overall Score</th>
                    <th>Faithfulness</th>
                    <th>Answer Relevancy</th>
                    <th>Context Precision</th>
                    <th>Context Recall</th>
                    <th>Chunks</th>
                    <th>Time (s)</th>
                </tr>
    """
    
    # Add rows to table
    best_strategy = results.get('best_strategy')
    for row in df.itertuples():
        row_class = 'best' if row.strategy == best_strategy else ''
        html += f"""
                <tr class="{row_class}">
                    <td>{row.name}</td>
                    <td><b>{row.overall_score:.3f}</b></td>
                    <td>{row.faithfulness:.3f}</td>
                    <td>{row.answer_relevancy:.3f}</td>
                    <td>{row.context_precision:.3f}</td>
                    <td>{row.context_recall:.3f}</td>
                    <td>{row.chunk_count}</td>
                    <td>{row.processing_time:.2f}</td>
                </tr>
        """
    
    # Add charts and recommendations
    html += f"""
            </table>
            
            <div class="chart">
                <h3>Overall Scores</h3>
                <img src="{base_filename}_overall_scores.png" alt="Overall Scores Chart" style="max-width:100%;">
            </div>
            
            <div class="chart">
                <h3>RAGAS Metrics Comparison</h3>
                <img src="{base_filename}_metrics_radar.png" alt="Metrics Radar Chart" style="max-width:100%;">
            </div>
            
            <div class="chart">
                <h3>Processing Time vs. Chunk Count</h3>
                <img src="{base_filename}_time_vs_chunks.png" alt="Processing Time vs Chunk Count" style="max-width:100%;">
            </div>
            
            <h2>Recommendations</h2>
            <div>
                <p>Based on RAGAS metrics evaluation, <b>{df.loc[df['strategy'] == best_strategy, 'name'].values[0]}</b> 
                is the best chunking strategy for PR #{results['pr_number']} with an overall score of 
                <b>{results['best_score']:.3f}</b>.</p>
                
                <h3>Strategy-specific observations:</h3>
                <ul>
                    <li><b>Semantic Chunking</b>: Preserves natural boundaries in code, good for well-structured PRs.</li>
                    <li><b>Hybrid Chunking</b>: Balances structure with size, versatile for mixed content PRs.</li>
                    <li><b>Fixed Size Chunking</b>: Provides consistent sizing, simple and predictable.</li>
                    <li><b>Hierarchical Chunking</b>: Maintains context hierarchies, good for complex nested structures.</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write HTML report
    with open(f"{output_dir}/{base_filename}_report.html", 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"HTML report saved to {output_dir}/{base_filename}_report.html")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
        if os.path.exists(json_path):
            visualize_chunking_comparison(json_path)
        else:
            print(f"File not found: {json_path}")
    else:
        # Find most recent comparison file
        chunking_dir = "chunking_results"
        if os.path.exists(chunking_dir):
            files = [os.path.join(chunking_dir, f) for f in os.listdir(chunking_dir) 
                   if f.startswith("chunking_comparison_") and f.endswith(".json")]
            if files:
                latest_file = max(files, key=os.path.getmtime)
                print(f"Using most recent comparison file: {latest_file}")
                visualize_chunking_comparison(latest_file)
            else:
                print(f"No chunking comparison files found in {chunking_dir}")
        else:
            print(f"Directory not found: {chunking_dir}")