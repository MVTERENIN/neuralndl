// Titanic Survival Classifier with TensorFlow.js
// Schema: Target: Survived (0/1). Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked.

// Global variables to store data and model
let trainData = null;
let testData = null;
let model = null;
let trainingHistory = null;
let validationData = null;
let validationLabels = null;
let validationPredictions = null;
let testPredictions = null;
let featureNames = [];

// DOM elements
const elements = {
    loadDataBtn: document.getElementById('load-data-btn'),
    preprocessBtn: document.getElementById('preprocess-btn'),
    createModelBtn: document.getElementById('create-model-btn'),
    trainBtn: document.getElementById('train-btn'),
    evaluateBtn: document.getElementById('evaluate-btn'),
    predictBtn: document.getElementById('predict-btn'),
    exportBtn: document.getElementById('export-btn'),
    saveModelBtn: document.getElementById('save-model-btn'),
    thresholdSlider: document.getElementById('threshold-slider'),
    thresholdValue: document.getElementById('threshold-value'),
    dataStatus: document.getElementById('data-status'),
    preprocessStatus: document.getElementById('preprocess-status'),
    modelStatus: document.getElementById('model-status'),
    trainingStatus: document.getElementById('training-status'),
    metricsStatus: document.getElementById('metrics-status'),
    predictionStatus: document.getElementById('prediction-status'),
    dataPreview: document.getElementById('data-preview'),
    preprocessInfo: document.getElementById('preprocess-info'),
    modelSummary: document.getElementById('model-summary'),
    accuracy: document.getElementById('accuracy'),
    precision: document.getElementById('precision'),
    recall: document.getElementById('recall'),
    f1: document.getElementById('f1')
};

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    // Set up event listeners
    elements.loadDataBtn.addEventListener('click', loadData);
    elements.preprocessBtn.addEventListener('click', preprocessData);
    elements.createModelBtn.addEventListener('click', createModel);
    elements.trainBtn.addEventListener('click', trainModel);
    elements.evaluateBtn.addEventListener('click', evaluateModel);
    elements.predictBtn.addEventListener('click', predictTestData);
    elements.exportBtn.addEventListener('click', exportPredictions);
    elements.saveModelBtn.addEventListener('click', saveModel);
    elements.thresholdSlider.addEventListener('input', updateThreshold);
});

// Load and inspect data from CSV files
async function loadData() {
    const trainFile = document.getElementById('train-file').files[0];
    const testFile = document.getElementById('test-file').files[0];
    
    if (!trainFile) {
        showError(elements.dataStatus, 'Please select a training CSV file');
        return;
    }
    
    try {
        showInfo(elements.dataStatus, 'Loading data...');
        
        // Load training data
        const trainText = await readFile(trainFile);
        trainData = parseCSV(trainText);
        
        // Load test data if provided
        if (testFile) {
            const testText = await readFile(testFile);
            testData = parseCSV(testText);
        }
        
        // Show data preview
        showDataPreview();
        
        // Analyze data
        analyzeData();
        
        // Enable next step
        elements.preprocessBtn.disabled = false;
        showSuccess(elements.dataStatus, 'Data loaded successfully!');
    } catch (error) {
        showError(elements.dataStatus, `Error loading data: ${error.message}`);
    }
}

// Read file as text
function readFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsText(file);
    });
}

// Parse CSV text to array of objects
function parseCSV(text) {
    const lines = text.split('\n').filter(line => line.trim() !== '');
    const headers = lines[0].split(',').map(header => header.trim());
    
    return lines.slice(1).map(line => {
        const values = line.split(',').map(value => value.trim());
        const row = {};
        headers.forEach((header, i) => {
            // Handle numeric values and missing data
            let value = values[i];
            if (value === '' || value === 'NA' || value === 'NaN') {
                row[header] = null;
            } else if (!isNaN(value) && value !== '') {
                row[header] = parseFloat(value);
            } else {
                row[header] = value;
            }
        });
        return row;
    });
}

// Show data preview in the UI
function showDataPreview() {
    let html = `<h3>Data Preview (First 5 Rows)</h3>`;
    html += `<p>Training Data: ${trainData.length} rows, ${Object.keys(trainData[0]).length} columns</p>`;
    
    if (testData) {
        html += `<p>Test Data: ${testData.length} rows, ${Object.keys(testData[0]).length} columns</p>`;
    }
    
    // Create table for training data preview
    html += `<h4>Training Data</h4><table><tr>`;
    const headers = Object.keys(trainData[0]);
    headers.forEach(header => {
        html += `<th>${header}</th>`;
    });
    html += `</tr>`;
    
    for (let i = 0; i < Math.min(5, trainData.length); i++) {
        html += `<tr>`;
        headers.forEach(header => {
            html += `<td>${trainData[i][header]}</td>`;
        });
        html += `</tr>`;
    }
    html += `</table>`;
    
    elements.dataPreview.innerHTML = html;
}

// Analyze data and show insights
function analyzeData() {
    // Calculate missing values
    const missingCount = {};
    const totalRows = trainData.length;
    
    Object.keys(trainData[0]).forEach(header => {
        missingCount[header] = trainData.filter(row => row[header] === null || row[header] === undefined).length;
    });
    
    // Show survival by sex and class
    const survivalBySex = {};
    const survivalByClass = {};
    
    trainData.forEach(row => {
        if (row.Sex && row.Survived !== null && row.Survived !== undefined) {
            if (!survivalBySex[row.Sex]) {
                survivalBySex[row.Sex] = { survived: 0, total: 0 };
            }
            survivalBySex[row.Sex].total++;
            if (row.Survived === 1) survivalBySex[row.Sex].survived++;
        }
        
        if (row.Pclass && row.Survived !== null && row.Survived !== undefined) {
            if (!survivalByClass[row.Pclass]) {
                survivalByClass[row.Pclass] = { survived: 0, total: 0 };
            }
            survivalByClass[row.Pclass].total++;
            if (row.Survived === 1) survivalByClass[row.Pclass].survived++;
        }
    });
    
    // Create visualization with tfjs-vis
    const survivalRateBySex = Object.keys(survivalBySex).map(sex => ({
        sex,
        rate: survivalBySex[sex].survived / survivalBySex[sex].total
    }));
    
    const survivalRateByClass = Object.keys(survivalByClass).map(pclass => ({
        pclass: `Class ${pclass}`,
        rate: survivalByClass[pclass].survived / survivalByClass[pclass].total
    }));
    
    // Create charts
    tfvis.render.barchart(
        { name: 'Survival Rate by Sex', tab: 'Data Analysis' },
        survivalRateBySex.map(item => ({ x: item.sex, y: item.rate })),
        { xAxisDomain: ['male', 'female'], yAxisDomain: [0, 1] }
    );
    
    tfvis.render.barchart(
        { name: 'Survival Rate by Class', tab: 'Data Analysis' },
        survivalRateByClass.map(item => ({ x: item.pclass, y: item.rate })),
        { yAxisDomain: [0, 1] }
    );
    
    // Show missing values
    let html = `<h3>Missing Values Analysis</h3><table>`;
    html += `<tr><th>Feature</th><th>Missing Count</th><th>Missing %</th></tr>`;
    
    Object.keys(missingCount).forEach(header => {
        const percent = ((missingCount[header] / totalRows) * 100).toFixed(2);
        html += `<tr><td>${header}</td><td>${missingCount[header]}</td><td>${percent}%</td></tr>`;
    });
    
    html += `</table>`;
    elements.dataPreview.innerHTML += html;
}

// Preprocess the data
function preprocessData() {
    try {
        showInfo(elements.preprocessStatus, 'Preprocessing data...');
        
        // Schema definition - change these for different datasets
        const target = 'Survived';
        const identifier = 'PassengerId';
        const numericFeatures = ['Age', 'Fare', 'SibSp', 'Parch'];
        const categoricalFeatures = ['Pclass', 'Sex', 'Embarked'];
        
        // Prepare training data
        const { features, labels, featureNames: names } = prepareData(
            trainData, 
            target, 
            identifier, 
            numericFeatures, 
            categoricalFeatures,
            true // isTraining
        );
        
        featureNames = names;
        
        // Store processed data as tensors
        const { dataTensor, labelTensor } = convertToTensors(features, labels);
        
        // Create validation split (80/20)
        const splitIndex = Math.floor(dataTensor.shape[0] * 0.8);
        const trainDataTensor = dataTensor.slice(0, splitIndex);
        const trainLabelTensor = labelTensor.slice(0, splitIndex);
        validationData = dataTensor.slice(splitIndex);
        validationLabels = labelTensor.slice(splitIndex);
        
        // Prepare test data if available
        if (testData) {
            const { features: testFeatures } = prepareData(
                testData, 
                null, // no target for test data
                identifier, 
                numericFeatures, 
                categoricalFeatures,
                false // isTraining
            );
            
            // Store test features
            testData = testFeatures;
        }
        
        // Update UI
        elements.createModelBtn.disabled = false;
        showSuccess(elements.preprocessStatus, 'Data preprocessing completed!');
        
        // Show preprocessing info
        elements.preprocessInfo.innerHTML = `
            <h3>Preprocessing Details</h3>
            <p>Training samples: ${trainDataTensor.shape[0]}</p>
            <p>Validation samples: ${validationData.shape[0]}</p>
            <p>Features: ${featureNames.join(', ')}</p>
            <p>Total features after encoding: ${featureNames.length}</p>
        `;
        
    } catch (error) {
        showError(elements.preprocessStatus, `Error preprocessing data: ${error.message}`);
    }
}

// Prepare data for training
function prepareData(data, target, identifier, numericFeatures, categoricalFeatures, isTraining) {
    const features = [];
    const labels = [];
    const featureNames = [];
    
    // Calculate imputation values from training data
    const imputationValues = {};
    
    if (isTraining) {
        numericFeatures.forEach(feature => {
            const values = data.map(row => row[feature]).filter(val => val !== null && val !== undefined);
            imputationValues[feature] = values.length > 0 ? 
                values.reduce((a, b) => a + b, 0) / values.length : 0;
        });
        
        categoricalFeatures.forEach(feature => {
            const values = data.map(row => row[feature]).filter(val => val !== null && val !== undefined);
            // Find mode
            const counts = {};
            let maxCount = 0;
            let mode = null;
            
            values.forEach(val => {
                counts[val] = (counts[val] || 0) + 1;
                if (counts[val] > maxCount) {
                    maxCount = counts[val];
                    mode = val;
                }
            });
            
            imputationValues[feature] = mode;
        });
    }
    
    // Process each row
    data.forEach(row => {
        const featureRow = [];
        
        // Add numeric features (with imputation)
        numericFeatures.forEach(feature => {
            let value = row[feature];
            if (value === null || value === undefined) {
                value = imputationValues[feature] || 0;
            }
            featureRow.push(value);
        });
        
        // Add categorical features (one-hot encoded)
        categoricalFeatures.forEach(feature => {
            let value = row[feature];
            if (value === null || value === undefined) {
                value = imputationValues[feature] || '';
            }
            
            // Get unique values for this feature (for one-hot encoding)
            // In a real implementation, we would pre-calculate these from training data
            const uniqueValues = ['male', 'female', 'S', 'C', 'Q', '1', '2', '3'];
            
            // One-hot encode
            uniqueValues.forEach(uniqueValue => {
                featureRow.push(value === uniqueValue ? 1 : 0);
            });
        });
        
        // Add engineered features
        // FamilySize = SibSp + Parch + 1
        const familySize = (row.SibSp || 0) + (row.Parch || 0) + 1;
        featureRow.push(familySize);
        
        // IsAlone = (FamilySize == 1)
        featureRow.push(familySize === 1 ? 1 : 0);
        
        features.push(featureRow);
        
        // Add target if this is training data
        if (isTraining && target) {
            labels.push(row[target]);
        }
    });
    
    // Build feature names
