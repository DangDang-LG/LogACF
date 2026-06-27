import io
import re
import os
import sys
import traceback
import numpy as np
import pandas as pd
from openai import OpenAI
from plot_well_predictions import plot_well_predictions  # 修改导入
from convert_PDF import convert_markdown_to_pdf
import requests
import anthropic
import base64
from plot_well_log_data import plot_well_log_data

client_chat_gpt = OpenAI(
    api_key="")
# Set the DeepSeek API key.
client_deepssek = OpenAI(
    api_key="",
    base_url="https://api.deepseek.com"
)

client_claude = anthropic.Anthropic(
    api_key=""
)

# Define the well logging analyst role.
Well_logging_analysis = {
    "role": "system",
    "content": (
        "You are a senior well logging analyst with expertise in logging interpretation and AI algorithm development.\n"
        "Your responsibility is to review the provided documents and break down the tasks into smaller subtasks.\n"
        "Then, you will delegate these subtasks to three specialized assistants:\n"
        " - Web_research_assistant: performs internet searches and summarizes results.\n"
        " - Code_programming_assistant: writes and improves Python code based on your instructions.\n"
        " - Report_generation_assistant: writes professional well-logging reports in Markdown format.\n"
        "When communicating with each assistant:\n"
        " 1. Provide clear, concise instructions.\n"
        " 2. Avoid unnecessary detail—focus on the objective and relevant context.\n"
        " 3. Summarize each subtask's goal and expected output.\n"
        "Your primary output should be a series of short, direct instructions to the appropriate assistant.\n"
        "If you need to revise or refine code or reports, clearly indicate the changes you want them to make.\n"
    )
}

# Define the report assistant role.
Report_generation_assistant = {
    "role": "assistant",
    "content": (
        "You are a specialized assistant for generating professional well logging interpretation reports.\n"
        "Your tasks:\n"
        "1. Wait for the user to provide background documents and data.\n"
        "2. Organize and integrate the given information into a comprehensive, detailed well logging report in Markdown format.\n"
        "3. Include relevant charts or data visualizations if asked.\n"
        "4. Only output your final report after receiving the explicit instruction 'please start'.\n"
        "   - Before that, you may collect or clarify information but do not provide partial or final reports.\n"
        "5. Ensure the final output is valid Markdown, ready for conversion to PDF.\n"
    )
}

# Define the Code_programming_assistant role
Code_programming_assistant = {
    "role": "assistant",
    "content": (
        "You are a Code_programming_assistant specialized in reservoir property prediction.\n"
        "🚨 **CRITICAL REQUIREMENT**: Every code solution MUST generate and save 'model_predictions.csv' file.\n\n"

        "📋 **MANDATORY OUTPUT FILE SPECIFICATIONS**:\n"
        "- File name: 'model_predictions.csv' (exact name)\n"
        "- Location: Results/{timestamp}/model_predictions.csv\n"
        "- Required columns: ['WELLNUM', 'DEPTH', 'PHIF_pred', 'SW_pred', 'VSH_pred']\n"

        "🔧 **CODE STRUCTURE REQUIREMENTS**:\n"
        "1. Always include create_results_folder() function\n"
        "2. Always include save_model_predictions() function\n"
        "3. Always call save_model_predictions() in main()\n"
        "4. Always print confirmation of successful file creation\n\n"

        "When you provide code:\n"
        "1. Begin with ```python and end with ```\n"
        "2. Follow with ```summary explaining the file generation approach ```\n"
        "3. Ensure every solution includes the mandatory file output\n"
    )
}

# Define the Web_research_assistant role
Web_research_assistant = {
    "role": "assistant",
    "content": (
        "You are a Web_research_assistant.\n"
        "You will be provided with documents or instructions about well logging.\n"
        "Your task is to:\n"
        "1. Identify relevant keywords and topics from the provided documents.\n"
        "2. Search the internet for credible information or data related to those topics.\n"
        "3. Summarize findings in a concise paragraph or list, focusing on direct answers and references where appropriate. Please do not hallucinate or fabricate data. \n"
        "4. Wait for the instruction 'please start' to provide your summarized findings.\n"
        "   - Before that, do not output any summaries.\n"
        "If asked, provide sources or URLs in plain text. Avoid personal opinions or speculation;\n"
        "only provide factual data from your research.\n"
    )
}


def remove_linebreaks(text):
    cleaned_text = text.replace("\n", " ").replace("\r", " ")
    return cleaned_text


def chat_with_gpt(messages, user_input):
    messages.append({"role": "user", "content": user_input})

    response = client_chat_gpt.chat.completions.create(
        model="gpt-4.1-2025-04-14",
        messages=messages
    )

    assistant_message = response.choices[0].message.content
    messages.append({"role": "assistant", "content": assistant_message})

    return assistant_message


def chat_with_deepseek(messages, user_input):
    messages.append({"role": "user", "content": user_input})

    response = client_deepssek.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
    )

    assistant_message = response.choices[0].message.content
    messages.append({"role": "assistant", "content": assistant_message})

    return assistant_message


def chat_with_claude(messages, user_input):
    system_message = None
    for msg in messages:
        if msg["role"] == "system":
            system_message = msg["content"]

    claude_messages = []
    for msg in messages:
        if msg["role"] == "system":
            continue
        elif msg["role"] == "user":
            claude_messages.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "assistant":
            claude_messages.append({"role": "assistant", "content": msg["content"]})

    claude_messages.append({"role": "user", "content": user_input})

    try:
        system_message = [system_message] if system_message else []

        response = client_claude.messages.create(
            model="claude-sonnet-4-6",
            system=system_message,
            messages=claude_messages,
            max_tokens=4000,
            temperature=0.3
        )

        if response.content and len(response.content) > 0:
            assistant_message = response.content[0].text
            messages.append({"role": "assistant", "content": assistant_message})
            return assistant_message
        else:
            return "Claude returned an empty response"

    except Exception as e:
        print(f"Error in Claude API call: {str(e)}")
        return str(e)


def create_vision_prompt_with_auto_analysis(well_name, auto_reservoir_data=None):
    """
    Create a vision analysis prompt with optional automatic reservoir identification results.

    Parameters:
        well_name (str): Name of the well.
        auto_reservoir_data (dict, optional): Automatically identified reservoir results.

    Returns:
        str: Constructed prompt string.
    """
    prompt = f"This is a well log comparison figure for **{well_name}**.\n\n"

    # 如果有自动储层识别结果，添加到提示词中
    if auto_reservoir_data and 'reservoir_zones' in auto_reservoir_data:
        prompt += "## 🤖 Automated Reservoir Identification Results (Reference)\n\n"
        prompt += "The following reservoir zones have been automatically identified using machine learning algorithms. "
        prompt += "Please use this information as a reference for your visual analysis:\n\n"

        zones = auto_reservoir_data['reservoir_zones']
        if zones:
            for i, zone in enumerate(zones, 1):
                prompt += f"**Zone R{i}** ({zone['quality']} Quality):\n"
                prompt += f"- Depth: {zone['start_depth']:.1f} - {zone['end_depth']:.1f} ft\n"
                prompt += f"- Thickness: {zone['thickness']:.1f} ft\n"
                prompt += f"- PHIF: {zone['avg_phif']:.3f}, SW: {zone['avg_sw']:.3f}, VSH: {zone['avg_vsh']:.3f}\n"
                prompt += f"- Oil Saturation: {zone['avg_so']:.3f}\n"
                prompt += f"- Net/Gross: {zone['net_gross_ratio']:.2f}\n\n"

            # 添加井总结信息
            if 'summary' in auto_reservoir_data:
                summary = auto_reservoir_data['summary']
                prompt += f"**Well Summary:**\n"
                prompt += f"- Total reservoir zones: {summary.get('total_reservoir_zones', 'N/A')}\n"
                prompt += f"- Total reservoir thickness: {summary.get('total_reservoir_thickness', 'N/A'):.1f} ft\n"
                prompt += f"- Reservoir density: {summary.get('reservoir_density', 'N/A'):.1f}%\n\n"
        else:
            prompt += "No reservoir zones were automatically identified for this well.\n\n"

        prompt += "---\n\n"

    prompt += (
        "## 📊 Your Visual Analysis Task:\n\n"
        "Please analyze the well log image and provide your expert interpretation:\n\n"
        "1. **Compare** your visual observations with the automated results above (if provided)\n"
        "2. **Identify** any additional reservoir zones that may have been missed\n"
        "3. **Validate** or challenge the automated identification results\n"
        "4. **Note** any discrepancies between automated and visual analysis\n\n"
        "For each zone you identify or validate, provide:\n"
        "- Top and bottom depth (in ft)\n"
        "- Estimated PHIF, SW, VSH values\n"
        "- Reservoir quality assessment\n"
        "- Agreement/disagreement with automated results\n\n"
        "🎯 Format your response using **Markdown**. Include well name and summary at the top.\n"
        "✅ Be sure your depth references match what's visible on the figure!\n"
        "⚠️  The automated results are for reference only - your expert interpretation is the final authority."
    )

    return prompt


def analyze_well_images_with_auto_reference(figure_info, reservoir_analysis_results,
                                            output_dir="vision_analysis_results"):
    """
    Perform visual analysis on well log images and incorporate automatic reservoir identification results as reference.

    Parameters:
        figure_info (list): List containing image metadata for each well.
        reservoir_analysis_results (dict): Enhanced reservoir analysis results including predicted zones and properties.
        output_dir (str): Directory path for saving analysis results.

    Returns:
        dict: Mapping from well name to vision analysis result {well_name: vision_result}.
    """

    print("\n" + "=" * 60)
    print("🔍 Starting integrated visual analysis with automated reservoir identification reference")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    well_to_vision_result = {}

    for fig in figure_info:
        match = re.search(r"Well_(\d+)", fig["location"])
        if not match:
            print(f"⚠️ Unable to extract well number from image path: {fig['location']}")
            continue

        well_num = match.group(1)
        well_name = f"Well {well_num}"
        image_path = fig["location"]

        print(f"\n📊 Processing {well_name}")
        print(f"📁 Image path: {image_path}")

        # Validate image file
        if not os.path.exists(image_path):
            print(f"❌ Image file does not exist: {image_path}")
            continue

        auto_reservoir_data = None

        if reservoir_analysis_results and "wells" in reservoir_analysis_results:
            auto_reservoir_data = reservoir_analysis_results["wells"].get(well_num, None)

            if auto_reservoir_data:
                zone_count = len(auto_reservoir_data.get("reservoir_zones", []))
                print(f"[INFO] Found automatic reservoir results: {zone_count} zones")
            else:
                print(f"[WARN] No reservoir results found for {well_name}")

        try:
            with open(image_path, "rb") as img_file:
                image_data = img_file.read()
                encoded = base64.b64encode(image_data).decode("utf-8")
                img_size = len(image_data) / 1024  # KB
            print(f"[INFO] Image loaded successfully. Size: {img_size:.1f} KB")

            if img_size > 5000:  # > 5MB
                print(f"[WARN] Large image detected ({img_size:.1f} KB), may affect performance")
        except Exception as e:
            print(f"[ERROR] Failed to read image: {str(e)}")
            traceback.print_exc()
            continue

        try:
            print(f"[STEP] Initializing vision analysis for {well_name}")
            prompt = create_vision_prompt_with_auto_analysis(well_name, auto_reservoir_data)
            client_claude_vision = anthropic.Anthropic(
                api_key=""
            )

            print(f"[RUN] Running integrated vision analysis for {well_name}")

            response = client_claude_vision.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4000,
                temperature=0.3,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": encoded
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )

            vision_result = response.content[0].text if response.content else "No response."

            if "Error" in vision_result or not vision_result.strip():
                print(f"[ERROR] Vision analysis returned invalid or empty result: {vision_result[:100]}...")
                continue

            result_file_path = os.path.join(output_dir, f"{well_name.replace(' ', '_')}_integrated_analysis.md")
            with open(result_file_path, "w", encoding="utf-8") as f:
                f.write(vision_result)
            print(f"💾 Integrated analysis results saved to: {result_file_path}")

            well_to_vision_result[well_name] = vision_result

            preview = vision_result[:300] + "..." if len(vision_result) > 300 else vision_result
            print(f"📝 Result preview:\n{preview}")

        except Exception as e:
            print(f"❌ Error during analysis of {well_name}: {str(e)}")
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(
        f"[STEP] Visual analysis pipeline completed successfully. Processed {len(well_to_vision_result)}/{len(figure_info)} wells")
    print(f"[INFO] Results directory: {os.path.abspath(output_dir)}")
    print("=" * 60)

    return well_to_vision_result


def read_input():
    print("Please enter the text (press enter twice to finish entering):")
    lines = []

    while True:
        line = input()
        if line == "":
            break
        lines.append(line)

    full_text = "\n".join(lines)
    cleaned_text = remove_linebreaks(full_text)
    return cleaned_text


def process_chatgpt_response(response):
    code, summary = extract_code_block(response)
    if code:
        file_name = save_code_to_file(code)
        result_code, result_analysis = run_code_file(file_name)
        return result_code, result_analysis, summary
    else:
        print("Code block not detected")
        return "", "", ""


def extract_code_block(response: str):
    """
    Extract Python code blocks and summary sections from the user response.

    1. Code block format: ```python ... ```
    2. Summary block format: ```summary ... ```

    If no corresponding content is detected, return a clear instruction message
    to guide the user to use the correct format.
    """
    match_code = re.search(r'```python(.*?)```', response, re.DOTALL)
    match_summary = re.search(r'```summary(.*?)```', response, re.DOTALL)


    if match_code:
        code_block = match_code.group(1).strip()
    else:
        code_block = (
            "No code detected.\n"
            "Please ensure your response contains a code block in the following format:\n"
            "```python\n"
            "# Your Python code here\n"
            "```\n"
        )


    if match_summary:
        summary_block = match_summary.group(1).strip()
    else:
        summary_block = (
            "No summary detected.\n"
            "Please ensure your response contains a summary block in the following format:\n"
            "```summary\n"
            "# Brief summary of the code\n"
            "```\n"
        )

    return code_block, summary_block


def save_code_to_file(code):
    folder_name = "generated_code"
    os.makedirs(folder_name, exist_ok=True)
    file_index = len(os.listdir(folder_name))
    file_name = os.path.join(folder_name, f"chatgpt_generated_code_{file_index + 1}.py")

    with open(file_name, 'w', encoding='utf-8') as file:
        file.write(code)
    print(f"The code has been saved to the file: {file_name}")

    return file_name


def run_code_file(file_name: str):
    output = io.StringIO()
    error_output = io.StringIO()

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = output
    sys.stderr = error_output

    default_message = (
        "Please avoid using logging.info for output.\n"
        "Use standard print or return statements instead.\n"
    )

    error_code = default_message
    error_analysis = default_message

    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            exec(file.read(), globals())

        result = output.getvalue()
        error = error_output.getvalue()

        if result:
            result_code = (
                "Execution completed successfully.\n"
                "Compare current results with previous outputs.\n"
                "If improvements are needed, provide updated code and summary.\n\n"
                "Requirements:\n"
                "- Python code must be wrapped in ```python```.\n"
                "- Summary must be wrapped in ```summary```.\n\n"
                "Standard output:\n\n" + result
            )

            result_analysis = "Execution result:\n" + result
            return result_code, result_analysis

    except Exception:
        error = traceback.format_exc()

        error_code = (
            "An error occurred during code execution.\n"
            "Please fix the issue and resend complete code.\n\n"
            "Required format:\n"
            "```python\n"
            "# corrected code\n"
            "```\n\n"
            "```summary\n"
            "# explanation\n"
            "```\n\n"
            "Traceback:\n" + error
        )

        error_analysis = "Execution error:\n" + error
        return error_code, error_analysis

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr

    return error_code, error_analysis


def read_file_contents(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return remove_linebreaks(file.read())


def read_initial_inputs(file_names):
    folder_path = "documents"
    contents = []

    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)

        if os.path.exists(file_path):
            contents.append(read_file_contents(file_path))
        else:
            print(f"[WARN] Missing file: {file_name}")

    return "\n".join(contents)



def detect_column_mapping(model_pred_data, real_test_data):
    """
    Automatically detect column mapping between prediction and ground truth data.
    """

    phif_variants = ['PHIF_pred', 'PHIF', 'Porosity_pred']
    sw_variants = ['SW_pred', 'SW', 'Water_Saturation_pred']
    vsh_variants = ['VSH_pred', 'VSH', 'Shale_Volume_pred']

    target_columns = ['PHIF', 'SW', 'VSH']

    if any(col not in real_test_data.columns for col in target_columns):
        print("[WARN] Missing target columns in reference data")
        return None

    column_mapping = {}

    pred_columns = list(model_pred_data.columns)

    phif_col = find_matching_column(pred_columns, phif_variants)
    sw_col = find_matching_column(pred_columns, sw_variants)
    vsh_col = find_matching_column(pred_columns, vsh_variants)

    if phif_col:
        column_mapping[phif_col] = 'PHIF'
    if sw_col:
        column_mapping[sw_col] = 'SW'
    if vsh_col:
        column_mapping[vsh_col] = 'VSH'

    if len(column_mapping) < 3:
        print(f"[WARN] Partial mapping found: {column_mapping}")
        return try_alternative_mapping(pred_columns, target_columns)

    return column_mapping


def find_matching_column(available_columns, variants):
    """
    Find a matching column name from available columns using candidate variants.

    Parameters:
        available_columns (list): List of available column names.
        variants (list): List of possible column name variants.

    Returns:
        str or None: Matched column name if found, otherwise None.
    """

    # Exact match search
    for variant in variants:
        if variant in available_columns:
            return variant

    # Case-insensitive matching
    for variant in variants:
        for col in available_columns:
            if variant.lower() == col.lower():
                return col

    return None


def try_alternative_mapping(pred_columns, target_columns):
    """
    Attempt alternative strategies for column name mapping.

    Parameters:
        pred_columns (list): Column names from prediction dataset.
        target_columns (list): Target standard column names.

    Returns:
        dict or None: Mapping dictionary if successful, otherwise None.
    """

    print("[INFO] Trying alternative column mapping strategy...")

    # Strategy 1: Direct matching
    direct_mapping = {}
    for target_col in target_columns:
        if target_col in pred_columns:
            direct_mapping[target_col] = target_col

    if len(direct_mapping) == 3:
        print(f"[INFO] Direct mapping used: {direct_mapping}")
        return direct_mapping

    # Strategy 2: Numeric column heuristic mapping
    numeric_columns = [
        col for col in pred_columns
        if col not in ['WELLNUM', 'DEPTH', 'Well', 'Depth']
    ]

    if len(numeric_columns) >= 3:
        print(f"[INFO] Found {len(numeric_columns)} numeric columns: {numeric_columns}")

        mapping = {}
        for i, target_col in enumerate(['PHIF', 'SW', 'VSH']):
            if i < len(numeric_columns):
                mapping[numeric_columns[i]] = target_col

        print(f"[INFO] Positional mapping applied: {mapping}")
        return mapping

    return None


def standardize_column_names(file_path, column_mapping):
    """
    Standardize prediction file column names to unified format:
    PHIF_pred, SW_pred, VSH_pred.

    Parameters:
        file_path (str): Path to CSV file.
        column_mapping (dict): Mapping {original_column: target_column}.

    Returns:
        bool: True if standardization succeeds, otherwise False.
    """

    try:
        df = pd.read_csv(file_path)
        original_columns = list(df.columns)

        rename_mapping = {}
        changes_needed = False

        # Build rename mapping
        for current_col, target_col in column_mapping.items():
            standard_col = f"{target_col}_pred"
            if current_col != standard_col:
                rename_mapping[current_col] = standard_col
                changes_needed = True

        if changes_needed:
            print(f"[INFO] Standardizing columns: {rename_mapping}")

            # Rename columns
            df.rename(columns=rename_mapping, inplace=True)

            # Backup original file
            backup_path = file_path.replace(".csv", "_backup.csv")
            pd.read_csv(file_path).to_csv(backup_path, index=False)
            print(f"[INFO] Backup saved to: {backup_path}")

            # Save standardized file
            df.to_csv(file_path, index=False)

            print(f"[INFO] File updated: {file_path}")
            print(f"[INFO] Columns: {original_columns} → {list(df.columns)}")

            return True

        else:
            print("[INFO] Columns already standardized")
            return True

    except Exception as e:
        print(f"[ERROR] Column standardization failed: {str(e)}")
        return False


def get_standard_column_mapping():
    """
    Return standardized column mapping for evaluation.

    Returns:
        dict: Standard mapping {'PHIF_pred': 'PHIF', 'SW_pred': 'SW', 'VSH_pred': 'VSH'}
    """
    return {
        "PHIF_pred": "PHIF",
        "SW_pred": "SW",
        "VSH_pred": "VSH"
    }

def find_best_rmse(
    predict_folder_path,
    rmse_thresholds=None,
    stop_on_guardrail=True
):
    """
    Enhanced function for finding the best RMSE model.
    It supports multiple prediction column name formats and automatic column standardization.

    Logic:
    1. Traverse all model_predictions.csv files.
    2. Calculate RMSE for each valid candidate model.
    3. Select the model with the lowest average RMSE.
    4. Apply the physical guardrail only to the selected best model.

    Parameters:
    predict_folder_path (str): Path to the prediction result folder.
    rmse_thresholds (dict, optional): RMSE thresholds for each target, for example:
        {
            'PHIF': 0.3,
            'SW': 0.3,
            'VSH': 0.3
        }
        If None, default thresholds will be used.
    stop_on_guardrail (bool): Whether to stop the workflow when the best model fails the guardrail.

    Returns:
    str: Path to the best model prediction file.
    """

    # Default thresholds, which can be adjusted according to the task requirements
    if rmse_thresholds is None:
        rmse_thresholds = {
            'PHIF': 0.3,
            'SW': 0.3,
            'VSH': 0.3
        }

    real_test_data = pd.read_csv('real_test_result.csv')
    print(f"Reference test data columns: {list(real_test_data.columns)}")
    print(f"Physical guardrail thresholds: {rmse_thresholds}")

    min_rmse = float('inf')
    best_folder = None
    best_model_pred_path = None
    best_rmse_values = None

    for root, dirs, files in os.walk(predict_folder_path):
        if 'model_predictions.csv' not in files:
            continue

        model_pred_path = os.path.join(root, 'model_predictions.csv')

        try:
            model_pred_data = pd.read_csv(model_pred_path)
            print(f"\nChecking file: {model_pred_path}")
            print(f"Prediction data columns: {list(model_pred_data.columns)}")

            # Automatically detect column name mapping
            column_mapping = detect_column_mapping(model_pred_data, real_test_data)

            if not column_mapping:
                print(f"⚠️ Unable to match prediction columns. Skipping file: {model_pred_path}")
                continue

            print(f"Using column mapping: {column_mapping}")

            # Standardize column names
            if standardize_column_names(model_pred_path, column_mapping):
                model_pred_data = pd.read_csv(model_pred_path)
                print(f"📁 Standardized columns: {list(model_pred_data.columns)}")
                column_mapping = get_standard_column_mapping()
            else:
                print(f"⚠️ Failed to standardize column names. Skipping file: {model_pred_path}")
                continue

            # Check whether all required columns are available
            missing_pred_cols = [
                pred_col for pred_col in column_mapping.keys()
                if pred_col not in model_pred_data.columns
            ]
            missing_true_cols = [
                true_col for true_col in column_mapping.values()
                if true_col not in real_test_data.columns
            ]

            if missing_pred_cols:
                print(f"⚠️ Missing prediction columns: {missing_pred_cols}")
                continue

            if missing_true_cols:
                print(f"⚠️ Missing reference columns: {missing_true_cols}")
                continue

            # =========================
            # Data length and depth consistency check
            # =========================
            if len(model_pred_data) != len(real_test_data):
                raise ValueError(
                    f"Data consistency check failed: prediction data and reference labels "
                    f"have inconsistent lengths. pred={len(model_pred_data)}, "
                    f"true={len(real_test_data)}"
                )

            # Check WELLNUM consistency
            if "WELLNUM" not in model_pred_data.columns or "WELLNUM" not in real_test_data.columns:
                raise ValueError(
                    "Data consistency check failed: missing required column 'WELLNUM' "
                    "in prediction data or reference labels."
                )

            if not np.all(model_pred_data["WELLNUM"].to_numpy() == real_test_data["WELLNUM"].to_numpy()):
                raise ValueError(
                    "Depth alignment check failed: inconsistent WELLNUM values between "
                    "prediction data and reference labels. This may indicate well mismatch "
                    "or incorrect sample ordering."
                )

            # Check DEPTH consistency
            if "DEPTH" not in model_pred_data.columns or "DEPTH" not in real_test_data.columns:
                raise ValueError(
                    "Depth alignment check failed: missing required column 'DEPTH' "
                    "in prediction data or reference labels."
                )

            pred_depth = model_pred_data["DEPTH"].to_numpy(dtype=float)
            true_depth = real_test_data["DEPTH"].to_numpy(dtype=float)

            if not np.allclose(pred_depth, true_depth, atol=1e-3, rtol=0):
                max_depth_diff = np.max(np.abs(pred_depth - true_depth))
                raise ValueError(
                    f"Depth alignment check failed: inconsistent DEPTH values between "
                    f"prediction data and reference labels. Maximum depth difference = "
                    f"{max_depth_diff:.6f}. This may indicate depth misalignment or "
                    f"incorrect sample ordering."
                )

            # Calculate RMSE
            rmse_values = {}
            for pred_col, true_col in column_mapping.items():
                pred_vals = model_pred_data[pred_col].to_numpy()
                true_vals = real_test_data[true_col].to_numpy()
                rmse = np.sqrt(np.mean((pred_vals - true_vals) ** 2))
                rmse_values[true_col] = rmse

            average_rmse = np.mean(list(rmse_values.values()))

            print(f"RMSE values: {rmse_values}")
            print(f"{root} average RMSE = {average_rmse:.6f}")

            # =========================
            # Only select the best model here.
            # Do NOT trigger the physical guardrail during traversal.
            # =========================
            if average_rmse < min_rmse:
                min_rmse = average_rmse
                best_folder = root
                best_model_pred_path = model_pred_path
                best_rmse_values = rmse_values

        except Exception as e:
            print(f"❌ Error while processing file {model_pred_path}: {str(e)}")
            continue

    # =========================
    # After traversing all candidates, check whether a best model was found
    # =========================
    if best_model_pred_path is None:
        print("❌ No valid model prediction file was found")
        raise FileNotFoundError(
            "No model file with valid prediction columns was found. Please check whether "
            "the model_predictions.csv files in the Results folder contain valid prediction columns."
        )

    print("\n" + "=" * 80)
    print("✅ Best model selected")
    print(f"✅ Best model path: {best_model_pred_path}")
    print(f"✅ Best folder: {best_folder}")
    print(f"✅ Best RMSE values: {best_rmse_values}")
    print(f"✅ Best average RMSE: {min_rmse:.6f}")
    print("=" * 80)

    # =========================
    # Physical guardrail check only for the best model
    # =========================
    guardrail_triggered = []

    for target_name, rmse_val in best_rmse_values.items():
        threshold = rmse_thresholds.get(target_name, None)
        if threshold is not None and rmse_val > threshold:
            guardrail_triggered.append(
                f"{target_name}: RMSE={rmse_val:.6f} > threshold={threshold:.6f}"
            )

    if guardrail_triggered:
        error_msg = (
            f"\n🚨 Physical guardrail triggered for the best model.\n"
            f"Best model file: {best_model_pred_path}\n"
            f"Best average RMSE: {min_rmse:.6f}\n"
            f"Best RMSE values: {best_rmse_values}\n"
            + "\n".join([f"   - {msg}" for msg in guardrail_triggered]) +
            "\n⛔ Workflow interrupted. The best available model still failed quality control. "
            "Please check data preprocessing, feature engineering, or model configuration."
        )

        print(error_msg)

        if stop_on_guardrail:
            raise RuntimeError(error_msg)
        else:
            print("⚠️ Best model failed the physical guardrail, but workflow will continue because stop_on_guardrail=False.")

    else:
        print("✅ Best model passed the physical guardrail check.")

    print('📋 The best model file has been standardized to: PHIF_pred, SW_pred, VSH_pred format')

    return best_model_pred_path



def format_reservoir_analysis_for_vision(reservoir_analysis_results):
    """
    Format reservoir analysis results into a structured text representation suitable for vision model input.

    Parameters:
        reservoir_analysis_results (dict): Enhanced reservoir analysis results containing well-level interpretations,
            reservoir zones, and associated petrophysical properties.

    Returns:
        str: Formatted text representation ready for vision model consumption.
    """

    if not reservoir_analysis_results or 'wells' not in reservoir_analysis_results:
        return "No automated reservoir analysis results available."

    formatted_text = "## 🤖 Automated Reservoir Identification Results\n\n"
    formatted_text += "The following reservoir zones have been automatically identified using advanced machine learning algorithms. "
    formatted_text += "These results should be used as a reference for expert interpretation:\n\n"

    wells = reservoir_analysis_results['wells']

    for well_num, well_data in wells.items():
        formatted_text += f"### Well {well_num}\n\n"

        # 添加井总结信息
        summary = well_data.get('summary', {})
        rmse = well_data.get('rmse_metrics', {})

        formatted_text += f"**Well Overview:**\n"
        formatted_text += f"- Total reservoir zones: {summary.get('total_reservoir_zones', 0)}\n"
        formatted_text += f"- Total reservoir thickness: {summary.get('total_reservoir_thickness', 0):.1f} ft\n"
        formatted_text += f"- Reservoir density: {summary.get('reservoir_density', 0):.1f}%\n"
        formatted_text += f"- Model accuracy (Avg RMSE): {rmse.get('avg_rmse', 'N/A'):.4f}\n\n"

        # 添加储层带详细信息
        zones = well_data.get('reservoir_zones', [])
        if zones:
            formatted_text += f"**Identified Reservoir Zones:**\n\n"
            for i, zone in enumerate(zones, 1):
                formatted_text += f"**Zone R{i}** ({zone['quality']} Quality):\n"
                formatted_text += f"- Depth interval: {zone['start_depth']:.1f} - {zone['end_depth']:.1f} ft\n"
                formatted_text += f"- Net thickness: {zone['thickness']:.1f} ft\n"
                formatted_text += f"- Average properties:\n"
                formatted_text += f"  - Porosity (PHIF): {zone['avg_phif']:.3f}\n"
                formatted_text += f"  - Water saturation (SW): {zone['avg_sw']:.3f}\n"
                formatted_text += f"  - Oil saturation (SO): {zone['avg_so']:.3f}\n"
                formatted_text += f"  - Shale volume (VSH): {zone['avg_vsh']:.3f}\n"
                formatted_text += f"- Reservoir quality metrics:\n"
                formatted_text += f"  - Net/Gross ratio: {zone['net_gross_ratio']:.2f}\n"
                formatted_text += f"  - Effective reservoir ratio: {zone['effective_ratio']:.2f}\n"
                formatted_text += f"  - Oil bearing ratio: {zone['oil_bearing_ratio']:.2f}\n\n"

            # 添加最佳储层信息
            best_reservoir = summary.get('best_reservoir')
            if best_reservoir:
                formatted_text += f"**Best Reservoir Zone:** R{best_reservoir['zone_number']} "
                formatted_text += f"({best_reservoir['quality']} quality, "
                formatted_text += f"RQI: {best_reservoir['reservoir_quality_index']:.4f})\n\n"
        else:
            formatted_text += f"No reservoir zones identified for this well based on current criteria.\n\n"

        formatted_text += "---\n\n"

    # 添加整体分析总结
    analysis_summary = reservoir_analysis_results.get('analysis_summary', {})
    formatted_text += f"### Overall Analysis Summary\n\n"
    formatted_text += f"- Total wells analyzed: {analysis_summary.get('total_wells', 0)}\n"
    formatted_text += f"- Total reservoir zones identified: {analysis_summary.get('total_reservoir_zones', 0)}\n"
    formatted_text += f"- Total reservoir thickness: {analysis_summary.get('total_reservoir_thickness', 0):.1f} ft\n"
    formatted_text += f"- Wells with reservoirs: {analysis_summary.get('wells_with_reservoirs', 0)}\n"
    formatted_text += f"- Average zones per well: {analysis_summary.get('average_zones_per_well', 0):.1f}\n\n"

    formatted_text += "**Note:** These automated results are based on machine learning models and should be validated "
    formatted_text += "through expert visual interpretation of the well log data.\n\n"

    return formatted_text


def generate_report_with_enhanced_analysis(
    conversation_history_report_gen,
    txt_data,
    conversation_history_code_results,
    figure_paths,
    report_guidelines,
    reservoir_analysis_results
):
    """
    Generate a comprehensive well logging interpretation report with enhanced
    reservoir analysis and multimodal vision integration.
    """

    try:
        print("\n" + "=" * 80)
        print("[STEP] Starting enhanced report generation pipeline")
        print("=" * 80)

        # =========================================================
        # Step 1: Provide documents and guidelines
        # =========================================================
        initial_prompt = (
            f"Documents:\n{txt_data}\n\n"
            f"Report Guidelines:\n{report_guidelines}\n\n"
            "Please acknowledge that you have received all materials."
        )

        response = chat_with_claude(conversation_history_report_gen, initial_prompt)
        print("[INFO] Documents and guidelines delivered to report agent")
        print(f"[Report Agent] {response}")

        # =========================================================
        # Step 2: Provide model execution results
        # =========================================================
        for entry in conversation_history_code_results:
            if entry["role"] == "user":
                model_results_prompt = (
                    "Model execution results and performance metrics:\n\n"
                    f"{entry['content']}\n\n"
                    "Please acknowledge receipt."
                )
                response = chat_with_claude(
                    conversation_history_report_gen,
                    model_results_prompt
                )

        print("[INFO] Model execution results delivered")
        print(f"[Report Agent] {response}")

        # =========================================================
        # Step 2.5: Provide reservoir analysis results
        # =========================================================
        if reservoir_analysis_results:
            reservoir_info_text = format_reservoir_analysis_for_vision(
                reservoir_analysis_results
            )

            reservoir_prompt = (
                "Enhanced automated reservoir analysis results:\n\n"
                f"{reservoir_info_text}\n\n"
                "Please incorporate this information into the report."
            )

            response = chat_with_claude(
                conversation_history_report_gen,
                reservoir_prompt
            )

            print("[INFO] Reservoir analysis results delivered")
            print(f"[Report Agent] {response}")

        # =========================================================
        # Step 3: Extract figure information
        # =========================================================
        figure_info = []
        picture_section = re.search(
            r'Here is the figure list:(.*?)---',
            txt_data,
            re.DOTALL
        )

        if picture_section:
            picture_text = picture_section.group(1)
            figure_matches = re.findall(
                r"Description: '([^']+)'\s+Location: '([^']+)'",
                picture_text,
                re.DOTALL
            )

            for desc, loc in figure_matches:
                figure_info.append({
                    "description": desc,
                    "location": loc.replace("\\", "/")
                })

        print("[INFO] Extracted figure list:")
        if figure_info:
            for fig in figure_info:
                print(f" - {fig['description']} @ {fig['location']}")
        else:
            print("[WARN] No figures matched parsing rules")

        # =========================================================
        # Step 4: Depth alignment from real data
        # =========================================================
        df = pd.read_csv("real_test_result.csv")

        well_depth_ranges = {
            str(int(well)): (
                int(group["DEPTH"].min()),
                int(group["DEPTH"].max())
            )
            for well, group in df.groupby("WELLNUM")
        }

        for fig in figure_info:
            match = re.search(r"Well_(\d+)", fig["location"])
            if match:
                well_num = match.group(1)
                if well_num in well_depth_ranges:
                    dmin, dmax = well_depth_ranges[well_num]
                    fig["description"] = (
                        f"Well {well_num} - Depth {dmin}-{dmax} ft "
                        f"- PHIF/SW/VSH comparison"
                    )

        # =========================================================
        # Step 5: Vision analysis with reservoir reference
        # =========================================================
        print("\n[STEP] Running enhanced vision analysis")

        well_to_vision_result = analyze_well_images_with_auto_reference(
            figure_info=figure_info,
            reservoir_analysis_results=reservoir_analysis_results,
            output_dir="enhanced_vision_analysis_results"
        )

        # =========================================================
        # Step 6: Provide figure instructions
        # =========================================================
        figure_content = (
            "Well logging figures (must be included in report):\n\n"
        )

        for fig in figure_info:
            figure_content += f"![{fig['description']}]({fig['location']})\n\n"

        response = chat_with_claude(
            conversation_history_report_gen,
            figure_content
        )

        print("[INFO] Figure instructions delivered")
        print(f"[Report Agent] {response}")

        # =========================================================
        # Step 7: Send all vision results
        # =========================================================
        print("[STEP] Sending vision analysis results")

        all_vision_analysis = (
            "Enhanced vision analysis results for all wells:\n\n"
        )

        for well_name, vision_result in well_to_vision_result.items():

            match = re.search(r'Well (\d+)', well_name)
            well_num = match.group(1) if match else "Unknown"

            depth_info = ""
            if well_num in well_depth_ranges:
                dmin, dmax = well_depth_ranges[well_num]
                depth_info = f" (Depth {dmin}-{dmax} ft)"

            auto_summary = ""
            if reservoir_analysis_results and "wells" in reservoir_analysis_results:
                well_data = reservoir_analysis_results["wells"].get(well_num)
                if well_data:
                    zone_count = len(well_data.get("reservoir_zones", []))
                    total_thickness = well_data.get("summary", {}).get(
                        "total_reservoir_thickness", 0
                    )
                    auto_summary = (
                        f" | Auto: {zone_count} zones, {total_thickness:.1f} ft"
                    )

            all_vision_analysis += (
                f"### {well_name}{depth_info}{auto_summary}\n\n"
                f"{vision_result}\n\n---\n\n"
            )

        vision_prompt = (
            all_vision_analysis +
            "\nIntegrate these results into the final report."
        )

        response = chat_with_claude(
            conversation_history_report_gen,
            vision_prompt
        )

        print("[INFO] Vision results delivered")

        # =========================================================
        # Step 8: Final report generation
        # =========================================================
        final_prompt = (
            "Generate complete well logging report using all inputs:\n"
            "1. Documents\n2. Model results\n3. Reservoir analysis\n"
            "4. Figures\n5. Vision analysis\n\n"
            "Return in markdown format."
        )

        final_report = chat_with_claude(
            conversation_history_report_gen,
            final_prompt
        )

        print("[SUCCESS] Report generation completed")

        # =========================================================
        # Step 9: Extract markdown
        # =========================================================
        markdown_content = extract_markdown_content(final_report)

        # =========================================================
        # Step 10: Append vision appendix
        # =========================================================
        vision_appendix = "\n\n## Appendix: Vision Analysis\n\n"

        for well_name, vision_result in well_to_vision_result.items():

            match = re.search(r'Well (\d+)', well_name)
            well_num = match.group(1) if match else "Unknown"

            image = None
            for fig in figure_info:
                if f"Well_{well_num}" in fig["location"]:
                    image = fig
                    break

            vision_appendix += f"### {well_name}\n\n"

            if image:
                vision_appendix += f"![{image['description']}]({image['location']})\n\n"

            vision_appendix += vision_result + "\n\n---\n\n"

        markdown_content += vision_appendix

        # =========================================================
        # Step 11: Save markdown
        # =========================================================
        md_path = "enhanced_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        print(f"[INFO] Markdown saved: {md_path}")

        # =========================================================
        # Step 12: Convert to PDF
        # =========================================================
        try:
            convert_markdown_to_pdf(
                md_path,
                "enhanced_report.pdf"
            )
            print("[INFO] PDF generated successfully")
        except Exception as e:
            print(f"[WARN] PDF conversion failed: {e}")

        # =========================================================
        # Step 13: Executive summary
        # =========================================================
        try:
            summary = generate_claude_vision_overall_summary(
                markdown_content,
                well_to_vision_result,
                reservoir_analysis_results
            )

            final_md = add_executive_summary_to_report(
                markdown_content,
                summary
            )

            with open(
                "final_report_with_summary.md",
                "w",
                encoding="utf-8"
            ) as f:
                f.write(final_md)

            print("[INFO] Final report with summary saved")

        except Exception as e:
            print(f"[WARN] Summary generation failed: {e}")

        return final_report

    except Exception as e:
        print(f"[ERROR] Report generation failed: {e}")
        traceback.print_exc()
        return str(e)

def generate_claude_vision_overall_summary(
    markdown_content,
    well_to_vision_result,
    reservoir_analysis_results
):
    """
    Generate an overall executive summary using Claude for multimodal well logging analysis.

    This function aggregates:
    - Full markdown report content
    - Per-well vision analysis results
    - Automated reservoir analysis results

    and sends them to a large language model to generate a high-level interpretation summary.

    Returns:
        str: Executive summary generated by Claude.
    """

    print("[STEP] Generating overall vision-based executive summary...")

    # =========================================================
    # Construct comprehensive context
    # =========================================================
    comprehensive_info = (
        "# Comprehensive Well Logging Analysis Summary\n\n"
        "## Report Content\n\n"
        f"{markdown_content}\n\n"
        "## Per-Well Vision Analysis\n\n"
    )

    for well_name, vision_result in well_to_vision_result.items():
        comprehensive_info += f"### {well_name}\n{vision_result}\n\n"

    # Add reservoir analysis if available
    if reservoir_analysis_results:
        comprehensive_info += (
            "## Automated Reservoir Analysis\n\n"
            + format_reservoir_analysis_for_vision(reservoir_analysis_results)
        )

    # =========================================================
    # Build LLM prompt
    # =========================================================
    summary_prompt = (
        "You are an expert petroleum reservoir interpretation assistant.\n\n"
        "Please provide a structured executive summary of this well logging project.\n\n"
        "Your response MUST include the following sections:\n\n"
        "1. Project Overview\n"
        "2. Key Geological and Petrophysical Findings\n"
        "3. Reservoir Quality and Distribution Assessment\n"
        "4. Model Performance Evaluation\n"
        "5. Development Recommendations\n\n"
        "Requirements:\n"
        "- Keep the summary concise (500–800 words)\n"
        "- Focus on engineering and geological insights\n"
        "- Avoid repetition of raw data\n"
        "- Emphasize actionable conclusions\n\n"
        "Input Data:\n"
        f"{comprehensive_info}"
    )

    try:
        # =========================================================
        # Call Claude
        # =========================================================
        conversation_history_summary = []

        overall_summary = chat_with_claude(
            conversation_history_summary,
            summary_prompt
        )

        print("[SUCCESS] Executive summary generated successfully")

        return overall_summary

    except Exception as e:
        print(f"[ERROR] Failed to generate executive summary: {str(e)}")

        return (
            "Executive summary generation failed due to technical issues."
        )

def add_executive_summary_to_report(markdown_content, executive_summary):
    """
    Insert an executive summary section at the beginning of a Markdown report.

    The function locates the first top-level heading (# Title) and inserts
    the executive summary immediately after it.

    Parameters:
        markdown_content (str): Original Markdown report content.
        executive_summary (str): Generated executive summary text.

    Returns:
        str: Updated Markdown content with executive summary inserted.
    """

    # Split markdown into lines
    lines = markdown_content.split('\n')
    insert_position = 0

    # Find first level-1 heading
    for i, line in enumerate(lines):
        if line.startswith('# '):
            insert_position = i + 1
            break

    # Construct summary section
    summary_section = [
        "",
        "## Executive Summary",
        "",
        executive_summary,
        "",
        "---",
        ""
    ]

    # Insert summary into original content
    final_lines = (
        lines[:insert_position]
        + summary_section
        + lines[insert_position:]
    )

    return '\n'.join(final_lines)


def extract_markdown_content(response):
    """Extract markdown content from Claude's response."""
    import re

    # Try to find markdown content between ```markdown and ``` tags
    match = re.search(r'```markdown(.*?)```', response, re.DOTALL)

    if match:
        return match.group(1).strip()
    else:
        # Check if the response already looks like markdown
        if re.search(r'#+\s+\w+|!\[.*?\]\(.*?\)', response):
            return response
        else:
            # Return the original content
            return response


def save_markdown(response, file_name):
    """
    Extract Markdown content from a model response and save it to a local file.

    Parameters:
        response (str): Response text from ChatGPT or other LLM.
        file_name (str): Output file path including filename and extension.
    """
    match = re.search(r'```markdown(.*?)```', response, re.DOTALL)
    if match:
        markdown_content = match.group(1).strip()
        with open(file_name, 'w', encoding='utf-8') as file:
            file.write(markdown_content)
        print(f"Markdown content saved to file: {file_name}")
    else:
        print("No Markdown content detected.")


def conversation_loop(conversation_history, response):
    while True:
        user_input = read_input()
        conversation_history.append({"role": "user", "content": user_input})
        response = chat_with_gpt(conversation_history, user_input)
        # print(f"Well_logging_analysis: {response}")

        if extract_code_block(response):
            print("Code block detected, enter loop.")
            print(f"\n Code_programming_assistant: {response}")
            return conversation_history, response


def manage_conversation_history(conversation_history_code):

    assistant_responses = [msg for msg in conversation_history_code if msg['role'] == 'assistant'][-2:]

    return assistant_responses


def detect_code_loop(
    conversation_history_code,
    response_code,
    conversation_history_analysis,
    n_iterations,
    conversation_history_report
):
    """
    Multi-agent iterative optimization loop for code generation, execution, analysis, and reporting.

    This function implements a closed-loop system involving three agents:

    - Code_programming_assistant: generates and updates code
    - Well_logging_analysis: provides domain-specific improvement suggestions
    - Report_generation_assistant: accumulates iteration-level results for final reporting

    The loop runs for n_iterations, where each iteration consists of:

    1. Parsing the code agent response into:
        - code_input: executable code or prompt
        - analysis_input: execution output or error logs
        - summary: functional description of the code

    2. Sending execution summary and results to the analysis agent to obtain refinement suggestions.

    3. Feeding analysis feedback back to the code agent for code refinement.

    4. Recording summary and execution results into the reporting agent for final report generation.

    Returns:
        tuple: Updated conversation histories for:
            - code agent
            - analysis agent
            - report agent
    """

    for i in range(n_iterations):

        # =====================================================
        # Step 1: Parse code agent output
        # =====================================================
        code_input, analysis_input, summary = process_chatgpt_response(response_code)

        # =====================================================
        # Step 2: Send results to domain analyst
        # =====================================================
        input_to_analysis = (
            "Code execution summary:\n" + summary + "\n\n"
            "Execution output:\n" + analysis_input + "\n\n"
            "Please provide concise improvement suggestions."
        )

        print("[STEP] Sending results to Well_logging_analysis")

        conversation_history_analysis.append({
            "role": "user",
            "content": input_to_analysis
        })

        response_analysis = chat_with_gpt(
            conversation_history_analysis,
            input_to_analysis
        )

        print(f"[ANALYST] {response_analysis}")

        # =====================================================
        # Step 3: Manage code history length
        # =====================================================
        conversation_history_code = manage_conversation_history(
            conversation_history_code
        )

        # =====================================================
        # Step 4: Send feedback back to code agent
        # =====================================================
        input_to_code = (
            code_input
            + "\n\n[Analysis Feedback]\n"
            + response_analysis
            + "\n\nConstraint: Avoid excessive computation methods (e.g., GridSearch)."
            + "Each run must generate and save: ./Results/{timestamp}/model_predictions.csv"
        )

        print("[STEP] Sending feedback to Code_programming_assistant")

        conversation_history_code.append({
            "role": "user",
            "content": input_to_code
        })

        response_code = chat_with_gpt(
            conversation_history_code,
            input_to_code
        )

        print(f"[CODE AGENT] {response_code}")

        # =====================================================
        # Step 5: Send iteration results to report agent
        # =====================================================
        report_content = (
            f"Iteration {i + 1} Results\n\n"
            f"Summary:\n{summary}\n\n"
            f"Execution Output:\n{analysis_input}\n"
        )

        conversation_history_report.append({
            "role": "user",
            "content": report_content
        })

        response_from_report = chat_with_gpt(
            conversation_history_report,
            report_content
        )

        print("[STEP] Report agent updated")

    return (
        conversation_history_code,
        conversation_history_analysis,
        conversation_history_report
    )


def main():
    """
    Main entry point of the multi-agent well logging analysis system.

    This pipeline integrates:
    - Web research agent
    - Well logging analysis agent
    - Code programming agent
    - Reservoir analysis module
    - Vision-enhanced report generation system
    """

    # =========================================================
    # Step 1: Initialize Well Logging Analysis Agent
    # =========================================================
    conversation_history_analysis = [Well_logging_analysis]

    initial_input = read_initial_inputs([
        "Tasks Description.txt",
        "Data Description.txt",
        "Knowledge.txt",
        "Task Requirement.txt",
        "Python Packages.txt",
        "Device Information.txt"
    ])

    conversation_history_analysis.append({
        "role": "user",
        "content": initial_input
    })

    response_analysis = chat_with_gpt(
        conversation_history_analysis,
        initial_input
    )

    print(f"[ANALYST INIT] {response_analysis}")

    # =========================================================
    # Step 2: Initialize Web Research Agent
    # =========================================================
    conversation_history_research = [Web_research_assistant]

    initial_input_research = read_initial_inputs([
        "Tasks Description.txt",
        "Data Description.txt",
        "Knowledge.txt"
    ])

    initial_input_research += (
        "\nPlease provide concise answers only."
        "\nStart now."
    )

    conversation_history_research.append({
        "role": "user",
        "content": initial_input_research
    })

    response_research = chat_with_gpt(
        conversation_history_research,
        initial_input_research
    )

    print(f"[RESEARCH AGENT] {response_research}")

    # ---------------------------------------------------------
    # Pass research output to analysis agent
    # ---------------------------------------------------------
    response_research_content = (
        "Web search results:\n"
        f"{response_research}\n\n"
        "Please generate instructions for Code_programming_assistant."
    )

    conversation_history_analysis.append({
        "role": "user",
        "content": response_research_content
    })

    response_analysis = chat_with_gpt(
        conversation_history_analysis,
        response_research_content
    )

    print(f"[ANALYST UPDATED] {response_analysis}")

    # =========================================================
    # Step 3: Initialize Code Programming Agent
    # =========================================================
    conversation_history_code = [Code_programming_assistant]

    initial_input_code = read_initial_inputs([
        "Tasks Description.txt",
        "Data Description.txt",
        "Knowledge.txt",
        "Task Requirement.txt",
        "Python Packages.txt",
        "Device Information.txt"
    ])

    conversation_history_code.append({
        "role": "user",
        "content": initial_input_code
    })

    response_code = chat_with_gpt(
        conversation_history_code,
        initial_input_code
    )

    print(f"[CODE INIT] {response_code}")

    # ---------------------------------------------------------
    # Code loop initialization
    # ---------------------------------------------------------
    conversation_history_code, response_code = conversation_loop(
        conversation_history_code,
        response_code
    )

    conversation_history_report = []

    # =========================================================
    # Step 4: Multi-iteration code optimization loop
    # =========================================================
    conversation_history_code, conversation_history_analysis, conversation_history_report = detect_code_loop(
        conversation_history_code=conversation_history_code,
        response_code=response_code,
        conversation_history_analysis=conversation_history_analysis,
        n_iterations=5,
        conversation_history_report=conversation_history_report
    )

    # =========================================================
    # Step 5: Reservoir analysis module
    # =========================================================
    print("\n" + "=" * 80)
    print("[STEP] Reservoir analysis started")
    print("=" * 80)

    try:
        best_model_pred_path = find_best_rmse('./Results')
    except FileNotFoundError as e:
        print(f"[ERROR] {str(e)}")
        print("[FATAL] No valid Results directory found")
        return None

    real_test_result_path = './real_test_result.csv'
    output_folder = './Results/Pictures'

    try:
        print("[STEP] Running reservoir analysis...")

        reservoir_analysis_results = plot_well_predictions(
            model_predictions_path=best_model_pred_path,
            real_test_result_path=real_test_result_path,
            output_folder=output_folder,
            options={
                "export_data": True,
                "dpi": 300,
                "min_thickness": 15,
                "max_gap": 5
            }
        )

        print("[SUCCESS] Reservoir analysis completed")

        summary = reservoir_analysis_results.get("analysis_summary", {})

        print("[INFO] Analysis summary:")
        print(f"  Wells: {summary.get('total_wells', 0)}")
        print(f"  Reservoir zones: {summary.get('total_reservoir_zones', 0)}")
        print(f"  Total thickness: {summary.get('total_reservoir_thickness', 0):.1f} ft")

    except Exception as e:
        print(f"[ERROR] Reservoir analysis failed: {str(e)}")
        return None

    # =========================================================
    # Step 6: Logging image generation
    # =========================================================
    try:
        gr_reservoir_well_reports, reservoir_output_text = plot_well_log_data(
            data_file="test.csv",
            output_dir=output_folder,
            identify_reservoirs=True
        )
        print("[SUCCESS] Log plots generated")
    except Exception as e:
        print(f"[WARN] Log plotting failed: {str(e)}")
        gr_reservoir_well_reports = {}

    # =========================================================
    # Step 7: Report generation inputs
    # =========================================================
    all_documents_content = read_initial_inputs([
        "Tasks Description.txt",
        "Data Description.txt",
        "Knowledge.txt",
        "Pictures.txt",
    ])

    report_guidelines = ""
    try:
        with open(os.path.join("memory_simplified", "Report.txt"), "r", encoding="utf-8") as f:
            report_guidelines = f.read()
        print("[SUCCESS] Report guidelines loaded")
    except Exception as e:
        print(f"[WARN] Failed to load report guidelines: {str(e)}")

    figure_paths = []
    try:
        for file in os.listdir(output_folder):
            if file.endswith("_comparison.png"):
                well_num = file.split("_")[1]
                figure_paths.append({
                    "path": os.path.join(output_folder, file),
                    "description": f"Well {well_num} results"
                })
    except Exception as e:
        print(f"[WARN] Failed to load figures: {str(e)}")

    # =========================================================
    # Step 8: Report generation agent
    # =========================================================
    conversation_history_report_gen = [Report_generation_assistant]

    try:
        print("[STEP] Generating final report")

        final_report = generate_report_with_enhanced_analysis(
            conversation_history_report_gen=conversation_history_report_gen,
            txt_data=all_documents_content,
            conversation_history_code_results=conversation_history_report,
            figure_paths=figure_paths,
            report_guidelines=report_guidelines,
            reservoir_analysis_results=reservoir_analysis_results
        )

        print("[SUCCESS] Report generation completed")

    except Exception as e:
        print(f"[ERROR] Report generation failed: {str(e)}")
        return None

    # =========================================================
    # Step 9: Final summary output
    # =========================================================
    print("\n" + "=" * 80)
    print("[SUCCESS] Pipeline execution completed")
    print("=" * 80)

    print("\n[FINAL SUMMARY]")

    for well_num, well_data in reservoir_analysis_results["wells"].items():
        zones = len(well_data.get("reservoir_zones", []))
        thickness = well_data.get("summary", {}).get("total_reservoir_thickness", 0)
        rmse = well_data.get("rmse_metrics", {}).get("avg_rmse", 0)

        print(f"Well {well_num}: {zones} zones, {thickness:.1f} ft, RMSE={rmse:.4f}")

    return reservoir_analysis_results


if __name__ == "__main__":
    main()