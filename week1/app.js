// Global variables to store data and charts
let mergedData = [];
let charts = {};

// Configuration - URLs for the datasets (can be swapped for other datasets)
const DATA_URLS = {
    train: "https://github.com/MVTERENIN/neuralndl/blob/main/week1/train.csv?raw=true",
    test: "https://github.com/MVTERENIN/neuralndl/blob/main/week1/test.csv?raw=true"
};

// Schema configuration - define features and their types
const SCHEMA = {
    target: 'Survived',  // The variable we're trying to predict
    features: {
        numeric: ['Age', 'Fare', 'SibSp', 'Parch'],
        categorical: ['Pclass', 'Sex', 'Embarked']
    },
    identifier: 'PassengerId'  // Exclude from analysis
};

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Set up event listeners for buttons
    document.getElementById('loadData').addEventListener('click', loadAndMergeData);
    document.getElementById('runEDA').addEventListener('click', runEDA);
    document.getElementById('exportCSV').addEventListener('click', exportCSV);
    document.getElementById('exportJSON').addEventListener('click', exportJSON);
});

// Load and merge the train and test datasets
function loadAndMergeData() {
    showMessage("Loading data...", "info");
    
    // Clear any existing data
    mergedData = [];
    
    // Load both datasets using PapaParse
    Promise.all([
        loadCSV(DATA_URLS.train, 'train'),
        loadCSV(DATA_URLS.test, 'test')
    ])
    .then(([trainData, testData]) => {
        // Add source column to distinguish between train and test data
        trainData.forEach(row => row.source = 'train');
        testData.forEach(row => row.source = 'test');
        
        // Merge the datasets
        mergedData = [...trainData, ...testData];
        
        showMessage(`Data loaded successfully. Total records: ${mergedData.length} (Train: ${trainData.length}, Test: ${testData.length})`, "success");
        
        // Update data overview
        updateDataOverview();
    })
    .catch(error => {
        showMessage(`Error loading data: ${error.message}`, "error");
        console.error(error);
    });
}

// Load a CSV file using PapaParse
function loadCSV(url, source) {
    return new Promise((resolve, reject) => {
        Papa.parse(url, {
            download: true,
            header: true,
            dynamicTyping: true,
            quotes: true,
            skipEmptyLines: true,
            complete: function(results) {
                if (results.errors.length > 0) {
                    reject(new Error(`Errors parsing ${source} data: ${results.errors.map(e => e.message).join(', ')}`));
                } else {
                    resolve(results.data);
                }
            },
            error: function(error) {
                reject(new Error(`Failed to load ${source} data from ${url}: ${error.message}`));
            }
        });
    });
}

// Update the data overview section
function updateDataOverview() {
    const overviewDiv = document.getElementById('dataOverview');
    const previewDiv = document.getElementById('previewTable');
    
    // Calculate dataset shape
    const trainCount = mergedData.filter(row => row.source === 'train').length;
    const testCount = mergedData.filter(row => row.source === 'test').length;
    
    // Create overview HTML
    overviewDiv.innerHTML = `
        <p><strong>Total Records:</strong> ${mergedData.length}</p>
        <p><strong>Train Records:</strong> ${trainCount}</p>
        <p><strong>Test Records:</strong> ${testCount}</p>
        <p><strong>Features:</strong> ${Object.keys(mergedData[0]).join(', ')}</p>
    `;
    
    // Create preview table (first 10 rows)
    if (mergedData.length > 0) {
        const headers = Object.keys(mergedData[0]);
        let tableHTML = `<table><thead><tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr></thead><tbody>`;
        
        // Add first 10 rows
        for (let i = 0; i < Math.min(10, mergedData.length); i++) {
            tableHTML += `<tr>${headers.map(h => `<td>${mergedData[i][h]}</td>`).join('')}</tr>`;
        }
        
        tableHTML += '</tbody></table>';
        previewDiv.innerHTML = tableHTML;
    }
}

// Run the full Exploratory Data Analysis
function runEDA() {
    if (mergedData.length === 0) {
        showMessage("Please load data first.", "error");
        return;
    }
    
    try {
        analyzeMissingValues();
        generateStatsSummary();
        createVisualizations();
        showMessage("EDA completed successfully.", "success");
    } catch (error) {
        showMessage(`Error during EDA: ${error.message}`, "error");
        console.error(error);
    }
}

// Analyze and visualize missing values
function analyzeMissingValues() {
    const columns = Object.keys(mergedData[0]);
    const missingData = [];
    
    // Calculate missing values for each column
    columns.forEach(column => {
        const missingCount = mergedData.filter(row => 
            row[column] === null || row[column] === undefined || row[column] === '' || isNaN(row[column])
        ).length;
        
        const missingPercent = (missingCount / mergedData.length) * 100;
        missingData.push({
            column: column,
            missing: missingCount,
            percent: missingPercent
        });
    });
    
    // Sort by missing percentage (descending)
    missingData.sort((a, b) => b.percent - a.percent);
    
    // Create bar chart of missing values
    const ctx = document.getElementById('missingValuesChart').getContext('2d');
    
    // Destroy previous chart if it exists
    if (charts.missingValues) {
        charts.missingValues.destroy();
    }
    
    charts.missingValues = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: missingData.map(item => item.column),
            datasets: [{
                label: 'Missing Values (%)',
                data: missingData.map(item => item.percent),
                backgroundColor: 'rgba(255, 99, 132, 0.6)',
                borderColor: 'rgba(255, 99, 132, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Percentage Missing'
                    }
                }
            }
        }
    });
}

// Generate statistical summary of the data
function generateStatsSummary() {
    const statsDiv = document.getElementById('statsSummary');
    let statsHTML = '';
    
    // Get train data for survival analysis
    const trainData = mergedData.filter(row => row.source === 'train');
    
    // Overall survival rate
    const survivedCount = trainData.filter(row => row.Survived === 1).length;
    const survivalRate = (survivedCount / trainData.length) * 100;
    
    statsHTML += `
        <h3>Overall Survival Rate</h3>
        <p><strong>Survived:</strong> ${survivedCount} (${survivalRate.toFixed(2)}%)</p>
        <p><strong>Did Not Survive:</strong> ${trainData.length - survivedCount} (${(100 - survivalRate).toFixed(2)}%)</p>
    `;
    
    // Numeric features summary
    statsHTML += '<h3>Numeric Features Summary</h3>';
    statsHTML += '<table><thead><tr><th>Feature</th><th>Mean</th><th>Median</th><th>Std Dev</th><th>Min</th><th>Max</th></tr></thead><tbody>';
    
    SCHEMA.features.numeric.forEach(feature => {
        const values = mergedData.map(row => row[feature]).filter(val => !isNaN(val));
        if (values.length > 0) {
            const mean = values.reduce((a, b) => a + b, 0) / values.length;
            const sorted = [...values].sort((a, b) => a - b);
            const median = sorted[Math.floor(sorted.length / 2)];
            const stdDev = Math.sqrt(values.reduce((sq, n) => sq + Math.pow(n - mean, 2), 0) / values.length);
            const min = Math.min(...values);
            const max = Math.max(...values);
            
            statsHTML += `<tr>
                <td>${feature}</td>
                <td>${mean.toFixed(2)}</td>
                <td>${median.toFixed(2)}</td>
                <td>${stdDev.toFixed(2)}</td>
                <td>${min.toFixed(2)}</td>
                <td>${max.toFixed(2)}</td>
            </tr>`;
        }
    });
    
    statsHTML += '</tbody></table>';
    
    // Categorical features summary
    statsHTML += '<h3>Categorical Features Summary</h3>';
    
    SCHEMA.features.categorical.forEach(feature => {
        statsHTML += `<h4>${feature} Distribution</h4>`;
        const counts = {};
        
        mergedData.forEach(row => {
            const value = row[feature] || 'Unknown';
            counts[value] = (counts[value] || 0) + 1;
        });
        
        statsHTML += '<table><thead><tr><th>Value</th><th>Count</th><th>Percentage</th></tr></thead><tbody>';
        
        for (const [value, count] of Object.entries(counts)) {
            const percentage = (count / mergedData.length) * 100;
            statsHTML += `<tr>
                <td>${value}</td>
                <td>${count}</td>
                <td>${percentage.toFixed(2)}%</td>
            </tr>`;
        }
        
        statsHTML += '</tbody></table>';
    });
    
    // Survival by feature (for train data only)
    if (trainData.length > 0) {
        statsHTML += '<h3>Survival Analysis by Feature (Train Data Only)</h3>';
        
        SCHEMA.features.categorical.forEach(feature => {
            statsHTML += `<h4>Survival by ${feature}</h4>`;
            const survivalRates = {};
            
            // Group by feature value
            trainData.forEach(row => {
                const value = row[feature] || 'Unknown';
                if (!survivalRates[value]) {
                    survivalRates[value] = { total: 0, survived: 0 };
                }
                survivalRates[value].total++;
                if (row.Survived === 1) {
                    survivalRates[value].survived++;
                }
            });
            
            statsHTML += '<table><thead><tr><th>Value</th><th>Total</th><th>Survived</th><th>Survival Rate</th></tr></thead><tbody>';
            
            for (const [value, data] of Object.entries(survivalRates)) {
                const rate = (data.survived / data.total) * 100;
                statsHTML += `<tr>
                    <td>${value}</td>
                    <td>${data.total}</td>
                    <td>${data.survived}</td>
                    <td>${rate.toFixed(2)}%</td>
                </tr>`;
            }
            
            statsHTML += '</tbody></table>';
        });
    }
    
    statsDiv.innerHTML = statsHTML;
}

// Create all visualizations
function createVisualizations() {
    createSurvivalChart('Sex', 'sexChart', 'Survival by Sex');
    createSurvivalChart('Pclass', 'classChart', 'Survival by Passenger Class');
    createDistributionChart('Age', 'ageChart', 'Age Distribution', 10);
    createDistributionChart('Fare', 'fareChart', 'Fare Distribution', 20);
    createEmbarkedChart();
    createCorrelationHeatmap();
}

// Create survival chart for a categorical feature
function createSurvivalChart(feature, canvasId, title) {
    const trainData = mergedData.filter(row => row.source === 'train');
    const categories = [...new Set(trainData.map(row => row[feature]))].filter(val => val !== undefined);
    
    const survivedData = [];
    const notSurvivedData = [];
    
    categories.forEach(category => {
        const categoryData = trainData.filter(row => row[feature] === category);
        const survivedCount = categoryData.filter(row => row.Survived === 1).length;
        const notSurvivedCount = categoryData.length - survivedCount;
        
        survivedData.push(survivedCount);
        notSurvivedData.push(notSurvivedCount);
    });
    
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    // Destroy previous chart if it exists
    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }
    
    charts[canvasId] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: categories,
            datasets: [
                {
                    label: 'Survived',
                    data: survivedData,
                    backgroundColor: 'rgba(75, 192, 192, 0.6)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                },
                {
                    label: 'Did Not Survive',
                    data: notSurvivedData,
                    backgroundColor: 'rgba(255, 99, 132, 0.6)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Count'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: title
                }
            }
        }
    });
}

// Create distribution chart for a numeric feature
function createDistributionChart(feature, canvasId, title, binCount = 10) {
    const values = mergedData.map(row => row[feature]).filter(val => !isNaN(val));
    const min = Math.min(...values);
    const max = Math.max(...values);
    const binSize = (max - min) / binCount;
    
    const bins = Array(binCount).fill(0);
    const labels = [];
    
    for (let i = 0; i < binCount; i++) {
        const binStart = min + i * binSize;
        const binEnd = binStart + binSize;
        labels.push(`${binStart.toFixed(1)} - ${binEnd.toFixed(1)}`);
        
        values.forEach(value => {
            if (value >= binStart && value < binEnd) {
                bins[i]++;
            }
        });
    }
    
    // Handle the last value inclusive
    values.forEach(value => {
        if (value === max) {
            bins[binCount - 1]++;
        }
    });
    
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    // Destroy previous chart if it exists
    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }
    
    charts[canvasId] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Frequency',
                data: bins,
                backgroundColor: 'rgba(54, 162, 235, 0.6)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Frequency'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: title
                    }
                }
            }
        }
    });
}

// Create embarked port chart
function createEmbarkedChart() {
    const embarkedCounts = {
        'C': { survived: 0, notSurvived: 0 },
        'Q': { survived: 0, notSurvived: 0 },
        'S': { survived: 0, notSurvived: 0 },
        'Unknown': { survived: 0, notSurvived: 0 }
    };
    
    const trainData = mergedData.filter(row => row.source === 'train');
    
    trainData.forEach(row => {
        const embarked = row.Embarked || 'Unknown';
        if (embarkedCounts[embarked]) {
            if (row.Survived === 1) {
                embarkedCounts[embarked].survived++;
            } else {
                embarkedCounts[embarked].notSurvived++;
            }
        }
    });
    
    const ctx = document.getElementById('embarkedChart').getContext('2d');
    
    // Destroy previous chart if it exists
    if (charts.embarkedChart) {
        charts.embarkedChart.destroy();
    }
    
    charts.embarkedChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Cherbourg (Survived)', 'Cherbourg (Not Survived)', 
                     'Queenstown (Survived)', 'Queenstown (Not Survived)',
                     'Southampton (Survived)', 'Southampton (Not Survived)',
                     'Unknown (Survived)', 'Unknown (Not Survived)'],
            datasets: [{
                data: [
                    embarkedCounts.C.survived, embarkedCounts.C.notSurvived,
                    embarkedCounts.Q.survived, embarkedCounts.Q.notSurvived,
                    embarkedCounts.S.survived, embarkedCounts.S.notSurvived,
                    embarkedCounts.Unknown.survived, embarkedCounts.Unknown.notSurvived
                ],
                backgroundColor: [
                    'rgba(75, 192, 192, 0.6)', 'rgba(75, 192, 192, 0.3)',
                    'rgba(54, 162, 235, 0.6)', 'rgba(54, 162, 235, 0.3)',
                    'rgba(255, 99, 132, 0.6)', 'rgba(255, 99, 132, 0.3)',
                    'rgba(255, 206, 86, 0.6)', 'rgba(255, 206, 86, 0.3)'
                ],
                borderColor: [
                    'rgba(75, 192, 192, 1)', 'rgba(75, 192, 192, 1)',
                    'rgba(54, 162, 235, 1)', 'rgba(54, 162, 235, 1)',
                    'rgba(255, 99, 132, 1)', 'rgba(255, 99, 132, 1)',
                    'rgba(255, 206, 86, 1)', 'rgba(255, 206, 86, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                },
                title: {
                    display: true,
                    text: 'Survival by Embarkation Port'
                }
            }
        }
    });
}

// Create correlation heatmap
function createCorrelationHeatmap() {
    // For simplicity, we'll use a subset of numeric features
    const features = ['Age', 'Fare', 'SibSp', 'Parch', 'Pclass'];
    const trainData = mergedData.filter(row => row.source === 'train');
    
    // Calculate correlation matrix
    const matrix = features.map(f1 => {
        return features.map(f2 => {
            if (f1 === f2) return 1;
            
            const values1 = trainData.map(row => row[f1]).filter(val => !isNaN(val));
            const values2 = trainData.map(row => row[f2]).filter(val => !isNaN(val));
            
            // Simple correlation calculation (Pearson's r)
            const n = Math.min(values1.length, values2.length);
            if (n === 0) return 0;
            
            const mean1 = values1.slice(0, n).reduce((a, b) => a + b, 0) / n;
            const mean2 = values2.slice(0, n).reduce((a, b) => a + b, 0) / n;
            
            let numerator = 0;
            let denom1 = 0;
            let denom2 = 0;
            
            for (let i = 0; i < n; i++) {
                numerator += (values1[i] - mean1) * (values2[i] - mean2);
                denom1 += Math.pow(values1[i] - mean1, 2);
                denom2 += Math.pow(values2[i] - mean2, 2);
            }
            
            const denominator = Math.sqrt(denom1 * denom2);
            return denominator === 0 ? 0 : numerator / denominator;
        });
    });
    
    const ctx = document.getElementById('correlationChart').getContext('2d');
    
    // Destroy previous chart if it exists
    if (charts.correlationChart) {
        charts.correlationChart.destroy();
    }
    
    charts.correlationChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: features,
            datasets: features.map((f, i) => {
                return {
                    label: f,
                    data: matrix[i],
                    backgroundColor: matrix[i].map(value => {
                        // Color based on correlation value
                        const intensity = Math.abs(value) * 255;
                        return value >= 0 
                            ? `rgba(0, 0, ${intensity}, 0.6)`
                            : `rgba(${intensity}, 0, 0, 0.6)`;
                    }),
                    borderColor: 'rgba(0, 0, 0, 0.2)',
                    borderWidth: 1
                };
            })
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Correlation Coefficient'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Feature Correlations'
                }
            }
        }
    });
}

// Export data as CSV
function exportCSV() {
    if (mergedData.length === 0) {
        showMessage("No data to export.", "error");
        return;
    }
    
    try {
        const csv = Papa.unparse(mergedData);
        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        
        link.setAttribute('href', url);
        link.setAttribute('download', 'titanic_merged_data.csv');
        link.style.visibility = 'hidden';
        
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        showMessage("CSV exported successfully.", "success");
    } catch (error) {
        showMessage(`Error exporting CSV: ${error.message}`, "error");
        console.error(error);
    }
}

// Export data summary as JSON
function exportJSON() {
    if (mergedData.length === 0) {
        showMessage("No data to export.", "error");
        return;
    }
    
    try {
        // Create a summary object
        const summary = {
            timestamp: new Date().toISOString(),
            totalRecords: mergedData.length,
            trainRecords: mergedData.filter(row => row.source === 'train').length,
            testRecords: mergedData.filter(row => row.source === 'test').length,
            features: Object.keys(mergedData[0]),
            survivalRate: calculateSurvivalRate()
        };
        
        const json = JSON.stringify(summary, null, 2);
        const blob = new Blob([json], { type: 'application/json;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        
        link.setAttribute('href', url);
        link.setAttribute('download', 'titanic_data_summary.json');
        link.style.visibility = 'hidden';
        
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        showMessage("JSON exported successfully.", "success");
    } catch (error) {
        showMessage(`Error exporting JSON: ${error.message}`, "error");
        console.error(error);
    }
}

// Calculate overall survival rate (train data only)
function calculateSurvivalRate() {
    const trainData = mergedData.filter(row => row.source === 'train');
    if (trainData.length === 0) return null;
    
    const survivedCount = trainData.filter(row => row.Survived === 1).length;
    return (survivedCount / trainData.length) * 100;
}

// Show message to user
function showMessage(message, type = "info") {
    const messageArea = document.getElementById('messageArea');
    const messageDiv = document.createElement('div');
    
    messageDiv.className = type === "error" ? "alert" : "success";
    messageDiv.textContent = message;
    
    messageArea.innerHTML = '';
    messageArea.appendChild(messageDiv);
    
    // Auto-remove success messages after 5 seconds
    if (type !== "error") {
        setTimeout(() => {
            messageDiv.remove();
        }, 5000);
    }
}
