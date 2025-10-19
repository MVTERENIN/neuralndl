// Student Employability Classifier - TensorFlow.js
// Dataset: Students' Employability Dataset - Philippines

// Global variables
let rawData = [];
let features = null;
let labels = null;
let model = null;
let trainingHistory = [];
let validationData = null;
let validationLabels = null;
let validationPredictions = null;

// Feature configuration - UPDATE THESE FOR DIFFERENT DATASETS
const FEATURE_COLUMNS = [
    'GENERAL APPEARANCE', 'MANNER OF SPEAKING', 'PHYSICAL CONDITION', 
    'MENTAL ALERTNESS', 'SELF-CONFIDENCE', 'ABILITY TO PRESENT IDEAS', 
    'COMMUNICATION SKILLS', 'Student Performance Rating'
];
const TARGET_COLUMN = 'CLASS'; // Target variable name

// Load dataset from GitHub
document.getElementById('loadData').addEventListener('click', async function() {
    try {
        const loadButton = this;
        loadButton.disabled = true;
        loadButton.textContent = 'Loading...';
        
        // Replace with your actual GitHub raw CSV URL
        const csvUrl = 'https://github.com/MVTERENIN/neuralndl/blob/main/NNDL_Project/data.csv';
        
        const response = await fetch(csvUrl);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        
        const csvText = await response.text();
        rawData = parseCSV(csvText);
        
        displayDataInfo(rawData);
        displayDataPreview(rawData);
        
        document.getElementById('preprocessData').disabled = false;
        
    } catch (error) {
        alert(`Error loading data: ${error.message}`);
        document.getElementById('loadData').disabled = false;
        document.getElementById('loadData').textContent = 'Load Dataset from GitHub';
    }
});

// Parse CSV text to array of objects
function parseCSV(csvText) {
    const lines = csvText.trim().split('\n');
    const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
    
    return lines.slice(1).map(line => {
        const values = line.split(',').map(v => v.trim());
        const obj = {};
        headers.forEach((header, index) => {
            obj[header] = values[index];
        });
        return obj;
    });
}

// Display dataset information
function displayDataInfo(data) {
    const infoDiv = document.getElementById('dataInfo');
    const missingCounts = calculateMissingValues(data);
    
    infoDiv.innerHTML = `
        <h3>Dataset Information</h3>
        <p><strong>Shape:</strong> ${data.length} rows × ${Object.keys(data[0]).length} columns</p>
        <p><strong>Target Variable:</strong> ${TARGET_COLUMN.toLowerCase()}</p>
        <p><strong>Features:</strong> ${FEATURE_COLUMNS.join(', ')}</p>
        <h4>Missing Values:</h4>
        <table>
            <tr><th>Column</th><th>Missing %</th></tr>
            ${Object.entries(missingCounts).map(([col, count]) => 
                `<tr><td>${col}</td><td>${((count/data.length)*100).toFixed(2)}%</td></tr>`
            ).join('')}
        </table>
    `;
}

// Calculate missing values percentage
function calculateMissingValues(data) {
    const missing = {};
    Object.keys(data[0]).forEach(col => {
        missing[col] = data.filter(row => !row[col] || row[col] === '').length;
    });
    return missing;
}

// Display data preview
function displayDataPreview(data) {
    const previewDiv = document.getElementById('dataPreview');
    const previewData = data.slice(0, 5);
    
    previewDiv.innerHTML = `
        <h3>Data Preview (first 5 rows)</h3>
        <table>
            <tr>${Object.keys(data[0]).map(col => `<th>${col}</th>`).join('')}</tr>
            ${previewData.map(row => 
                `<tr>${Object.values(row).map(val => `<td>${val}</td>`).join('')}</tr>`
            ).join('')}
        </table>
    `;
}

// Preprocess data
document.getElementById('preprocessData').addEventListener('click', function() {
    try {
        const processed = preprocessData(rawData);
        features = processed.features;
        labels = processed.labels;
        
        document.getElementById('preprocessInfo').innerHTML = `
            <p><strong>Preprocessing Complete</strong></p>
            <p>Features shape: [${features.shape[0]}, ${features.shape[1]}]</p>
            <p>Labels shape: [${labels.shape[0]}]</p>
            <p>Class distribution: ${processed.classDistribution.employable} Employable, 
            ${processed.classDistribution.unemployable} Unemployable</p>
        `;
        
        // Create visualization
        createDataVisualization(processed.classDistribution);
        
        document.getElementById('createModel').disabled = false;
        
    } catch (error) {
        alert(`Error preprocessing data: ${error.message}`);
    }
});

// Preprocess the dataset
function preprocessData(data) {
    const featuresArray = [];
    const labelsArray = [];
    let employableCount = 0;
    let unemployableCount = 0;

    data.forEach(row => {
        // Extract features (ignore case in column names)
        const featureRow = FEATURE_COLUMNS.map(col => {
            const value = row[col.toLowerCase()];
            if (!value || isNaN(value)) throw new Error(`Invalid feature value in ${col}: ${value}`);
            return parseFloat(value);
        });
        featuresArray.push(featureRow);

        // Extract and encode target variable
        const targetValue = row[TARGET_COLUMN.toLowerCase()];
        if (!targetValue) throw new Error(`Missing target value in ${TARGET_COLUMN}`);
        
        // Convert to binary (1 for Employable, 0 for Unemployable)
        const encodedLabel = targetValue.toLowerCase().includes('employable') ? 1 : 0;
        labelsArray.push(encodedLabel);
        
        if (encodedLabel === 1) employableCount++;
        else unemployableCount++;
    });

    return {
        features: tf.tensor2d(featuresArray),
        labels: tf.tensor1d(labelsArray),
        classDistribution: {
            employable: employableCount,
            unemployable: unemployableCount
        }
    };
}

// Create data visualization
function createDataVisualization(classDistribution) {
    const data = [
        { index: 'Employable', value: classDistribution.employable },
        { index: 'Unemployable', value: classDistribution.unemployable }
    ];

    const surface = { name: 'Class Distribution', tab: 'Data Analysis' };
    tfvis.render.barchart(surface, data, {
        xLabel: 'Class',
        yLabel: 'Count',
        width: 400,
        height: 300
    });
}

// Create model
document.getElementById('createModel').addEventListener('click', function() {
    try {
        model = createModel();
        document.getElementById('modelInfo').innerHTML = `
            <p><strong>Model Created Successfully</strong></p>
            <p>Architecture: Input(${FEATURE_COLUMNS.length}) → Dense(16, relu) → Dense(1, sigmoid)</p>
        `;
        
        // Print model summary
        model.summary();
        
        document.getElementById('trainModel').disabled = false;
        
    } catch (error) {
        alert(`Error creating model: ${error.message}`);
    }
});

// Create the neural network model
function createModel() {
    const model = tf.sequential({
        layers: [
            // Single hidden layer with 16 neurons and ReLU activation
            tf.layers.dense({
                inputShape: [FEATURE_COLUMNS.length],
                units: 16,
                activation: 'relu'
            }),
            // Output layer with sigmoid activation for binary classification
            tf.layers.dense({
                units: 1,
                activation: 'sigmoid'
            })
        ]
    });

    // Compile the model
    model.compile({
        optimizer: 'adam',
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });

    return model;
}

// Train model
document.getElementById('trainModel').addEventListener('click', async function() {
    try {
        const trainButton = this;
        trainButton.disabled = true;
        trainButton.textContent = 'Training...';
        
        // Split data into training and validation sets (80/20)
        const splitIndex = Math.floor(features.shape[0] * 0.8);
        
        const trainFeatures = features.slice(0, splitIndex);
        const trainLabels = labels.slice(0, splitIndex);
        validationData = features.slice(splitIndex);
        validationLabels = labels.slice(splitIndex);
        
        // Train the model
        await trainModel(model, trainFeatures, trainLabels, validationData, validationLabels);
        
        // Enable prediction and evaluation
        document.getElementById('predictNew').disabled = false;
        document.getElementById('thresholdSlider').disabled = false;
        trainButton.textContent = 'Training Complete';
        
    } catch (error) {
        alert(`Error training model: ${error.message}`);
        document.getElementById('trainModel').disabled = false;
        document.getElementById('trainModel').textContent = 'Train Model';
    }
});

// Train the model
async function trainModel(model, trainFeatures, trainLabels, valFeatures, valLabels) {
    const surface = { name: 'Training Metrics', tab: 'Training' };
    
    // Early stopping callback
    let bestValLoss = Infinity;
    let patienceCounter = 0;
    const patience = 5;
    
    const history = await model.fit(trainFeatures, trainLabels, {
        epochs: 50,
        batchSize: 32,
        validationData: [valFeatures, valLabels],
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                // Early stopping logic
                if (logs.val_loss < bestValLoss) {
                    bestValLoss = logs.val_loss;
                    patienceCounter = 0;
                } else {
                    patienceCounter++;
                }
                
                if (patienceCounter >= patience) {
                    console.log(`Early stopping at epoch ${epoch + 1}`);
                    model.stopTraining = true;
                }
                
                // Update training history
                trainingHistory.push({
                    epoch: epoch + 1,
                    loss: logs.loss,
                    accuracy: logs.acc,
                    val_loss: logs.val_loss,
                    val_acc: logs.val_acc
                });
                
                // Plot to tfvis
                tfvis.show.fitCallbacks(surface, ['loss', 'val_loss', 'acc', 'val_acc']);
            }
        }
    });
    
    return history;
}

// Make predictions on validation set and calculate metrics
document.getElementById('thresholdSlider').addEventListener('input', function() {
    const threshold = parseFloat(this.value);
    document.getElementById('thresholdValue').textContent = threshold.toFixed(2);
    
    if (validationData && model) {
        evaluateModel(threshold);
    }
});

// Evaluate model with given threshold
async function evaluateModel(threshold) {
    try {
        // Make predictions on validation set
        const predictions = model.predict(validationData);
        const probs = await predictions.data();
        predictions.dispose();
        
        // Convert probabilities to binary predictions using threshold
        const binaryPreds = probs.map(p => p >= threshold ? 1 : 0);
        const trueLabels = await validationLabels.data();
        
        // Calculate metrics
        const metrics = calculateMetrics(trueLabels, binaryPreds);
        
        // Update ROC curve if not already calculated
        if (!validationPredictions) {
            validationPredictions = probs;
            plotROCCurve(trueLabels, probs);
        }
        
        // Display metrics
        displayMetrics(metrics, threshold);
        
    } catch (error) {
        console.error('Error evaluating model:', error);
    }
}

// Calculate classification metrics
function calculateMetrics(trueLabels, predictions) {
    let tp = 0, fp = 0, tn = 0, fn = 0;
    
    for (let i = 0; i < trueLabels.length; i++) {
        if (trueLabels[i] === 1 && predictions[i] === 1) tp++;
        else if (trueLabels[i] === 0 && predictions[i] === 1) fp++;
        else if (trueLabels[i] === 0 && predictions[i] === 0) tn++;
        else if (trueLabels[i] === 1 && predictions[i] === 0) fn++;
    }
    
    const accuracy = (tp + tn) / (tp + fp + tn + fn);
    const precision = tp / (tp + fp) || 0;
    const recall = tp / (tp + fn) || 0;
    const f1 = 2 * (precision * recall) / (precision + recall) || 0;
    
    return { tp, fp, tn, fn, accuracy, precision, recall, f1 };
}

// Display metrics and confusion matrix
function displayMetrics(metrics, threshold) {
    const confusionDiv = document.getElementById('confusionMatrix');
    const metricsDiv = document.getElementById('metricsDisplay');
    
    confusionDiv.innerHTML = `
        <h4>Confusion Matrix (Threshold: ${threshold.toFixed(2)})</h4>
        <table>
            <tr><th></th><th>Predicted Negative</th><th>Predicted Positive</th></tr>
            <tr><th>Actual Negative</th><td>${metrics.tn}</td><td>${metrics.fp}</td></tr>
            <tr><th>Actual Positive</th><td>${metrics.fn}</td><td>${metrics.tp}</td></tr>
        </table>
    `;
    
    metricsDiv.innerHTML = `
        <h4>Performance Metrics</h4>
        <p><strong>Accuracy:</strong> ${(metrics.accuracy * 100).toFixed(2)}%</p>
        <p><strong>Precision:</strong> ${(metrics.precision * 100).toFixed(2)}%</p>
        <p><strong>Recall:</strong> ${(metrics.recall * 100).toFixed(2)}%</p>
        <p><strong>F1-Score:</strong> ${(metrics.f1 * 100).toFixed(2)}%</p>
    `;
}

// Plot ROC curve and calculate AUC
function plotROCCurve(trueLabels, probabilities) {
    // Calculate ROC curve points
    const thresholds = Array.from({ length: 100 }, (_, i) => i / 100);
    const rocPoints = thresholds.map(threshold => {
        const binaryPreds = probabilities.map(p => p >= threshold ? 1 : 0);
        const metrics = calculateMetrics(trueLabels, binaryPreds);
        return {
            fpr: metrics.fp / (metrics.fp + metrics.tn) || 0,
            tpr: metrics.tp / (metrics.tp + metrics.fn) || 0,
            threshold
        };
    });
    
    // Calculate AUC (Area Under Curve)
    let auc = 0;
    for (let i = 1; i < rocPoints.length; i++) {
        auc += (rocPoints[i].fpr - rocPoints[i-1].fpr) * 
               (rocPoints[i].tpr + rocPoints[i-1].tpr) / 2;
    }
    
    // Plot ROC curve
    const rocData = {
        values: rocPoints.map(point => ({ x: point.fpr, y: point.tpr })),
        series: ['ROC Curve']
    };
    
    const surface = { name: 'ROC Curve', tab: 'Evaluation' };
    tfvis.render.linechart(surface, rocData, {
        xLabel: 'False Positive Rate',
        yLabel: 'True Positive Rate',
        width: 400,
        height: 400
    });
    
    // Display AUC
    document.getElementById('rocCurve').innerHTML += `
        <p><strong>AUC:</strong> ${auc.toFixed(3)}</p>
    `;
}

// Make predictions on new data
document.getElementById('predictNew').addEventListener('click', function() {
    const resultsDiv = document.getElementById('predictionResults');
    resultsDiv.innerHTML = `
        <p><strong>Prediction functionality ready.</strong></p>
        <p>To make predictions on new data, you would need to preprocess the new data using the same pipeline and call model.predict().</p>
        <p><em>Note: For a complete deployment, you would add file upload for new CSV data here.</em></p>
    `;
});

// Initialize application
console.log('Student Employability Classifier initialized');
console.log('To use with different datasets, update FEATURE_COLUMNS and TARGET_COLUMN in app.js');
