#!/bin/bash




set -e  


DATA_DIR="data"
OUTPUT_DIR="conv_output/convergence"
MAX_CHAINS=8
MIN_SAMPLES=50
BURNIN_FRAC=0.25


MODELS=(
    "m1vexp"
    "m1vphi2"
    "m1vphi24"
    "m2vexp"
    "m2vphi2"
    "m2vphi24"
    "m3vexp"
    "m3vphi2"
    "m3vphi24"
)


RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' 

echo -e "${BLUE}=== MCMC Convergence Analysis for All Models ===${NC}"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Max chains per model: $MAX_CHAINS"
echo "Min samples per chain: $MIN_SAMPLES"
echo "Burn-in fraction: $BURNIN_FRAC"
echo ""


mkdir -p "$OUTPUT_DIR"


declare -a SUCCESS_MODELS=()
declare -a FAILED_MODELS=()
declare -a SKIPPED_MODELS=()


for model in "${MODELS[@]}"; do
    echo -e "${YELLOW}Processing model: $model${NC}"
    
    
    sql_file="${DATA_DIR}/${model}.sql"
    if [[ ! -f "$sql_file" ]]; then
        echo -e "${RED}  ⚠ Skipping $model: SQL file not found ($sql_file)${NC}"
        SKIPPED_MODELS+=("$model")
        echo ""
        continue
    fi
    
    
    if uv run python convergence_analysis.py \
        --model "$model" \
        --data_dir "$DATA_DIR" \
        --max_chains "$MAX_CHAINS" \
        --min_samples "$MIN_SAMPLES" \
        --burnin_frac "$BURNIN_FRAC" \
        --outdir "$OUTPUT_DIR"; then
        
        echo -e "${GREEN}  ✓ Success: $model${NC}"
        SUCCESS_MODELS+=("$model")
    else
        echo -e "${RED}  ✗ Failed: $model${NC}"
        FAILED_MODELS+=("$model")
    fi
    
    echo ""
done


echo -e "${BLUE}=== Summary ===${NC}"
echo -e "${GREEN}Successful models (${#SUCCESS_MODELS[@]}):${NC}"
for model in "${SUCCESS_MODELS[@]}"; do
    echo "  ✓ $model"
done

if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
    echo -e "${RED}Failed models (${#FAILED_MODELS[@]}):${NC}"
    for model in "${FAILED_MODELS[@]}"; do
        echo "  ✗ $model"
    done
fi

if [[ ${#SKIPPED_MODELS[@]} -gt 0 ]]; then
    echo -e "${YELLOW}Skipped models (${#SKIPPED_MODELS[@]}):${NC}"
    for model in "${SKIPPED_MODELS[@]}"; do
        echo "  ⚠ $model (no data file)"
    done
fi

echo ""
    echo -e "${BLUE}Output files saved to: $OUTPUT_DIR${NC}"
    echo "Files generated:"
    ls -la "$OUTPUT_DIR"/*.png 2>/dev/null | wc -l | xargs echo "  Total plots:"
    
    
    echo -e "${BLUE}Creating comprehensive R-hat table...${NC}"
    if uv run python convergence_analysis.py \
        --model "m1vexp" \
        --data_dir "$DATA_DIR" \
        --max_chains "$MAX_CHAINS" \
        --min_samples "$MIN_SAMPLES" \
        --burnin_frac "$BURNIN_FRAC" \
        --outdir "$OUTPUT_DIR" \
        --create_table; then
        
        echo -e "${GREEN}✓ R-hat table created successfully${NC}"
        ls -la "$OUTPUT_DIR"/all_models_rhat_table.png 2>/dev/null | head -1 | awk '{print "  PNG Table: " $9}'
        ls -la "$OUTPUT_DIR"/all_models_rhat_table.csv 2>/dev/null | head -1 | awk '{print "  CSV Table: " $9}'
    else
        echo -e "${RED}✗ Failed to create R-hat table${NC}"
    fi


index_file="${OUTPUT_DIR}/index.html"
echo -e "${BLUE}Generating HTML index: $index_file${NC}"

cat > "$index_file" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>MCMC Convergence Analysis Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f8f9fa; }
        .header { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .model { margin-bottom: 30px; background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .model h2 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; margin-top: 0; }
        .rhat-section { background: #f8f9fa; padding: 15px; border-radius: 6px; margin: 15px 0; border-left: 4px solid #007bff; }
        .rhat-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        .rhat-table th, .rhat-table td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #ddd; }
        .rhat-table th { background: #f1f3f4; font-weight: bold; }
        .rhat-excellent { color: #28a745; font-weight: bold; }
        .rhat-warning { color: #fd7e14; font-weight: bold; }
        .rhat-poor { color: #dc3545; font-weight: bold; }
        .status-badge { padding: 3px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; text-transform: uppercase; }
        .badge-excellent { background: #d4edda; color: #155724; }
        .badge-warning { background: #fff3cd; color: #856404; }
        .badge-poor { background: #f8d7da; color: #721c24; }
        .plots { display: flex; flex-wrap: wrap; gap: 20px; margin-top: 20px; }
        .plot { border: 1px solid #eee; padding: 15px; border-radius: 6px; background: #fff; }
        .plot img { max-width: 400px; height: auto; border-radius: 4px; }
        .plot h4 { margin-top: 0; color: #495057; }
        .summary { background: #e3f2fd; padding: 15px; border-radius: 6px; margin-bottom: 20px; }
        .interpretation { background: #fff3e0; padding: 15px; border-radius: 6px; margin-top: 20px; border-left: 4px solid #ff9800; }
    </style>
</head>
<body>
    <div class="header">
        <h1>MCMC Convergence Analysis Results</h1>
        <p><strong>Generated:</strong> $(date)</p>
        <div class="summary">
            <h3>Summary</h3>
            <p><strong>Models analyzed:</strong> ${#SUCCESS_MODELS[@]}</p>
            <p><strong>Interpretation:</strong> R-hat values close to 1.0 indicate good convergence. Values &lt; 1.05 are excellent, &lt; 1.1 acceptable, &gt; 1.1 indicate convergence issues.</p>
        </div>
    </div>
EOF

for model in "${SUCCESS_MODELS[@]}"; do
    cat >> "$index_file" << EOF
    <div class="model">
        <h2>Model: $model</h2>
        
        <div class="rhat-section">
            <h3>Gelman-Rubin R-hat Diagnostics</h3>
            <div id="rhat-$model">Loading R-hat values...</div>
        </div>
        
        <h3>Diagnostic Plots</h3>
        <div class="plots">
EOF
    
    
    for plot in "${OUTPUT_DIR}/${model}"_*.png; do
        if [[ -f "$plot" ]]; then
            plot_name=$(basename "$plot")
            plot_title=$(echo "$plot_name" | sed 's/_/ /g' | sed 's/.png//' | sed 's/\b\w/\U&/g')
            cat >> "$index_file" << EOF
            <div class="plot">
                <h4>$plot_title</h4>
                <img src="$plot_name" alt="$plot_name">
            </div>
EOF
        fi
    done
    
    cat >> "$index_file" << EOF
        </div>
    </div>
EOF
done

cat >> "$index_file" << 'EOF'
    <div class="interpretation">
        <h3>Interpretation Guide</h3>
        <p><strong>R-hat values:</strong> Measure convergence of MCMC chains. Values close to 1.0 indicate good mixing.</p>
        <p><strong>Trace plots:</strong> Show parameter evolution over iterations. Should look like "white noise" after burn-in.</p>
        <p><strong>Prior vs Posterior:</strong> If posterior differs substantially from uniform prior, results are data-driven rather than prior-dominated.</p>
    </div>

    <script>
        // Load R-hat data for each model
        const models = [EOF

for model in "${SUCCESS_MODELS[@]}"; do
    cat >> "$index_file" << EOF
'$model',EOF
done

cat >> "$index_file" << 'EOF'
];
        
        models.forEach(model => {
            fetch(`${model}_rhat.json`)
                .then(response => response.json())
                .then(data => {
                    const container = document.getElementById(`rhat-${model}`);
                    if (data.parameters) {
                        let html = '<table class="rhat-table"><thead><tr><th>Parameter</th><th>R-hat Value</th><th>Status</th></tr></thead><tbody>';
                        
                        data.parameters.forEach((param, i) => {
                            const rhat = data.rhat_values[i];
                            const status = data.convergence_status[i];
                            const rhatClass = `rhat-${status}`;
                            const badgeClass = `badge-${status}`;
                            
                            html += `<tr>
                                <td><strong>${param}</strong></td>
                                <td><span class="${rhatClass}">${rhat.toFixed(3)}</span></td>
                                <td><span class="status-badge ${badgeClass}">${status}</span></td>
                            </tr>`;
                        });
                        
                        html += '</tbody></table>';
                        container.innerHTML = html;
                    } else {
                        container.innerHTML = '<p>R-hat data not available</p>';
                    }
                })
                .catch(error => {
                    console.error(`Error loading R-hat data for ${model}:`, error);
                    document.getElementById(`rhat-${model}`).innerHTML = '<p>Error loading R-hat data</p>';
                });
        });
    </script>
</body>
</html>
EOF

echo -e "${GREEN}✓ HTML index generated: $index_file${NC}"
echo ""


if [[ ${#FAILED_MODELS[@]} -eq 0 ]]; then
    echo -e "${GREEN}🎉 All available models processed successfully!${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠ Some models failed. Check the output above for details.${NC}"
    exit 1
fi