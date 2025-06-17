## 🤖InsightX : Data Analysis Agent 🤖

An intelligent data analysis agent powered by Mixtral-8x7B that helps you analyze datasets, generate visualizations, and provide insights through natural language interaction.

## Key Features ✨

- **Natural Language Processing**: Interact with your data using plain English
- **Intelligent Data Analysis**: Get detailed insights and statistical analysis
- **Automated Visualization**: Generate charts and graphs based on your queries
- **Context Management**: Maintains conversation history and dataset context
- **Session Management**: Save and load analysis sessions
- **Smart Data Type Inference**: Automatically recommends optimal data types for your datasets
- **Multi-Format Support**: Process various file types including:
  - Documents (PDF, DOCX)
  - Spreadsheets (CSV, Excel)
  - Images (PNG, JPG)
  - Text files (TXT)

## Setup 🛠️

1. Clone the repository:
```bash
git clone https://github.com/UjjWaL-0911/InsightX-An-AI-Data-Analyst.git
cd InsightX-An-AI-Data-Analyst
```

2. Create required directories:(This Repo already contains the folders)
```bash
mkdir Graph_Plots Saved_Sessions Datasets Configs
```

3. Set up environment variables:
- Create a `.env` file in the `Configs` directory
- Add your Together AI API key:
```
TOGETHER_API_KEY=your_api_key_here
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure 📁

```
data-analyst-agent/
├── Configs/
│   └── .env
├── Datasets/
├── Graph_Plots/
├── Saved_Sessions/
├── agent_core.py
├── file_processor.py
├── data_preprocessor.py
├── test_agent.py
├── app.py
├── requirements.txt
└── README.md

```

## Usage 🚀

1. For CLI Testing:
```bash
python test_agent.py
```
2. For GUI Testing:
```bash
streamlit run app.py
```
## Supported File Types 📄

The agent can process various file formats:

### Documents
- PDF files (`.pdf`)
- Microsoft Word documents (`.docx`)

### Spreadsheets
- CSV files (`.csv`)
- Excel files (`.xlsx`, `.xls`)

### Images
- PNG images (`.png`)
- JPEG images (`.jpg`, `.jpeg`)

### Text
- Plain text files (`.txt`)

Each file type is processed appropriately:
- Documents are parsed for text content and structure
- Spreadsheets are converted to pandas DataFrames
- Images are analyzed for visual content
- Text files are processed for natural language analysis

## Inference Configuration ⚙️

The agent uses Mixtral-8x7B model with the following default parameters:
- Model: `mistralai/Mixtral-8x7B-Instruct-v0.1`
- Max Tokens: 1024
- Temperature: 0.7
- Top P: 0.7
- Top K: 50
- Repetition Penalty: 1.1

## Key Components 🔑

1. **FileProcessor**   : Handles dataset loading and preprocessing
2. **DataPreprocessor**: Handles data cleaning i.e., missing values,conversion to appropriate datatype,duplicate values
3. **ContextManager**  : Manages dataset state and conversation history
4. **LLMInterface**    : Handles communication with the Mixtral-8x7B model
5. **IntelligentAgent**: Main class that orchestrates the analysis workflow


## Contributing 🤝

Contributions are welcome! Please feel free to raise issues and submit a Pull Request.

## Acknowledgments 🙏

- Powered by [Together AI](https://www.together.ai/) and Mixtral-8x7B
- Built with Python, pandas,matplotlib and Streamlit 