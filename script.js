// --- Configuration ---
// Replace this with your actual Firebase API URL
const API_ENDPOINT = "http://127.0.0.1:5000/data";
const REFRESH_INTERVAL_MS = 5000; // 5 seconds

// API Endpoints for Historical Data (Synchronized with app.py)
const HISTORY_ENDPOINTS = {
    rainfall: "http://127.0.0.1:5000/api/rainfall",
    waterlevel: "http://127.0.0.1:5000/api/waterlevel",
    temperature: "http://127.0.0.1:5000/api/temperature",
    flowrate: "http://127.0.0.1:5000/api/flowrate" // New Endpoint
};

// Mock Geocoding data for demonstration
const MOCK_LOCATIONS = {
    "agartala": [23.8314, 91.2868],
    "zone 1": [23.85, 91.30],
    "zone 2": [23.80, 91.25],
    "zone 3": [23.88, 91.20],
    "nit agartala": [23.8314, 91.2868],
    "delhi": [28.7041, 77.1025]
};

let rainChartInstance;
let waterChartInstance;
let fullRainChartInstance;
let fullWaterChartInstance;
let fullTempChartInstance;
let fullFlowChartInstance; // New Chart Instance
let searchMarker; // Marker for user-searched location
let currentData; // Global variable to hold the latest fetched data
let refreshIntervalId; // To store the ID of the polling interval
let currentPage = ''; // To track the currently rendered page

// --- Dark Mode Logic ---

function initializeDarkMode() {
    const toggleButton = document.getElementById('darkModeToggle');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    // 1. Check local storage for user preference
    let isDarkMode = localStorage.getItem('darkMode') === 'true';

    // 2. If no preference, use system preference
    if (localStorage.getItem('darkMode') === null) {
        isDarkMode = prefersDark;
    }

    setDarkMode(isDarkMode);

    toggleButton.addEventListener('click', () => {
        const currentMode = document.body.classList.contains('dark-mode');
        setDarkMode(!currentMode);
    });
}

function setDarkMode(isDark) {
    const toggleButton = document.getElementById('darkModeToggle');
    if (isDark) {
        document.body.classList.add('dark-mode');
        localStorage.setItem('darkMode', 'true');
        toggleButton.innerText = '☀️ Light Mode';
    } else {
        document.body.classList.remove('dark-mode');
        localStorage.setItem('darkMode', 'false');
        toggleButton.innerText = '🌙 Dark Mode';
    }
}
// --- Utility Functions ---

function updateLastUpdateTime() {
    const now = new Date();
    document.getElementById('last-update').innerText = `Last Update: ${now.toLocaleTimeString()}`;
}

function updateSystemStatus(riskLevel) {
    const statusElement = document.getElementById('system-status');
    statusElement.className = 'status-indicator'; // Reset classes

    if (riskLevel === "High") {
        statusElement.classList.add('status-critical');
        statusElement.innerText = 'System Status: CRITICAL ALERT';
    } else if (riskLevel === "Moderate") {
        statusElement.classList.add('status-alert');
        statusElement.innerText = 'System Status: ALERT';
    } else {
        statusElement.classList.add('status-ok');
        statusElement.innerText = 'System Status: OK';
    }
}

// --- Data Rendering Functions ---
function getPredictedLevel(data) {
    // Directly use the new field from the backend, which is either a number or null
    return data.ai_predicted_level_cm !== undefined ? data.ai_predicted_level_cm : null;
}

function renderDataCards(data) {
    const dataElements = [
        { id: "rain", value: data.rainfall_mm.toFixed(1) },
        { id: "water", value: data.water_level_cm.toFixed(1) },
        { id: "temp", value: data.temperature_c.toFixed(1) },
        { id: "flow", value: data.flow_rate_mlpm.toFixed(1) } // Updated Flow Rate Card (ml/min)
    ];

    // 1. Update standard data cards
    dataElements.forEach(item => {
        const element = document.getElementById(item.id);
        if (element && element.innerText !== item.value) {
            // Apply animation class if data has changed
            element.classList.remove('animate-update');
            // Force reflow/repaint to restart animation
            void element.offsetWidth;
            element.classList.add('animate-update');
        }
        if (element) element.innerText = item.value;
    });

    // 2. Update Prediction Panel
    const risk = data.ai_risk || 'Low';
    const riskStatusElement = document.getElementById("ai-risk-status");
    const forecastMetricElement = document.getElementById("ai-forecast-metric");
    const recommendationMsgElement = document.getElementById("ai-recommendation-msg");
    const predictionCard = document.getElementById("prediction-card");

    // Set risk status text and class
    if (riskStatusElement) {
        riskStatusElement.innerText = risk;
        riskStatusElement.className = `risk-status risk-${risk.toLowerCase()}`;
    }

    // Apply risk class to the main panel for background color
    if (predictionCard) {
        predictionCard.classList.remove('risk-low', 'risk-moderate', 'risk-high');
        predictionCard.classList.add(`risk-${risk.toLowerCase()}`);
    }

    // Extract predicted water level using helper function
    const rawPredictedLevel = getPredictedLevel(data);
    const predictedLevelText = rawPredictedLevel !== null ? `${rawPredictedLevel.toFixed(1)} cm` : '--';

    if (forecastMetricElement) {
        forecastMetricElement.innerText = predictedLevelText;
    }

    if (recommendationMsgElement) {
        recommendationMsgElement.innerText = data.ai_recommendation || 'Awaiting analysis...';
    }

    // 3. Update system status indicator
    updateSystemStatus(risk);
}

function renderAlerts(alerts) {
    const tableBody = document.getElementById("alertTable");
    tableBody.innerHTML = ''; // Clear existing alerts

    if (alerts.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="4">No active alerts.</td></tr>';
        return;
    }

    alerts.forEach(a => {
        const row = `
            <tr>
                <td>${a.time}</td>
                <td>${a.zone}</td>
                <td>${a.message}</td>
                <td><span class="risk-tag risk-tag-${a.risk.toLowerCase()}">${a.risk}</span></td>
            </tr>
        `;
        tableBody.innerHTML += row;
    });
}

// --- Chart Functions ---

function createChartOptions(unit, displayLegend = false) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            y: {
                beginAtZero: true,
                title: {
                    display: true,
                    text: unit
                },
                ticks: {
                    callback: function(value) {
                        return value + unit;
                    }
                }
            }
        },
        plugins: {
            legend: {
                display: displayLegend
            }
        }
    };
}

function createChart(elementId, type, label, data, labels, borderColor, backgroundColor, unit = '') {
    const ctx = document.getElementById(elementId);
    if (!ctx) return null; // Return null if element doesn't exist (page not loaded)

    const options = createChartOptions(unit, label.includes('Temperature')); // Display legend for temperature chart

    return new Chart(ctx.getContext('2d'), {
        type: type,
        data: {
            labels: labels,
            datasets: [{
                label: label,
                data: data,
                borderColor: borderColor,
                backgroundColor: backgroundColor,
                tension: 0.3,
                fill: true
            }]
        },
        options: options
    });
}

function initializeDashboardCharts(history, current) {
    // Destroy existing instances if they exist (important for dynamic loading)
    if (rainChartInstance) rainChartInstance.destroy();
    if (waterChartInstance) waterChartInstance.destroy();

    // Rainfall Chart (Dashboard)
    rainChartInstance = createChart(
        'rainChart', 'line', 'Rainfall (mm)', history.rainfall, history.labels,
        'rgba(0, 123, 255, 1)', 'rgba(0, 123, 255, 0.2)', 'mm'
    );

    // Water Level Chart (Dashboard) - Enhanced with Prediction
    const predictedLevel = getPredictedLevel(current);
    const waterLabels = [...history.labels];
    const waterData = [...history.water_level];

    if (predictedLevel !== null) {
        // Add 'Forecast' label and a null point followed by the prediction value
        waterLabels.push('Forecast');
        waterData.push(null, predictedLevel);
    }

    waterChartInstance = createChart(
        'waterChart', 'line', 'Water Level (cm)', waterData, waterLabels,
        'rgba(40, 167, 69, 1)', 'rgba(40, 167, 69, 0.2)', 'cm'
    );

    // Add a second dataset for the prediction line (dotted line)
    if (predictedLevel !== null) {
        const predictionDataset = Array(history.water_level.length).fill(null);
        // The prediction line starts at the last historical point and goes to the forecast point
        predictionDataset.push(history.water_level[history.water_level.length - 1], predictedLevel);

        waterChartInstance.data.datasets.push({
            label: 'Predicted Level',
            data: predictionDataset,
            borderColor: 'rgba(255, 99, 132, 1)', // Red/Pink for prediction
            backgroundColor: 'transparent',
            borderDash: [5, 5], // Dotted line
            pointRadius: [0, 0, 5], // Only show point on the forecast value
            pointBackgroundColor: 'rgba(255, 99, 132, 1)',
            tension: 0.3,
            fill: false
        });
        waterChartInstance.update();
    }
}

function updateDashboardCharts(history, current) {
    if (rainChartInstance) {
        rainChartInstance.data.labels = history.labels;
        rainChartInstance.data.datasets[0].data = history.rainfall;
        rainChartInstance.update();
    }

    if (waterChartInstance) {
        const predictedLevel = getPredictedLevel(current);
        const waterLabels = [...history.labels];
        const waterData = [...history.water_level];
        
        // Reset datasets to handle dynamic prediction presence
        waterChartInstance.data.datasets = [];

        if (predictedLevel !== null) {
            waterLabels.push('Forecast');
            waterData.push(null, predictedLevel);
        }

        // 1. Main Water Level Dataset
        waterChartInstance.data.labels = waterLabels;
        waterChartInstance.data.datasets.push({
            label: 'Water Level (cm)',
            data: waterData,
            borderColor: 'rgba(40, 167, 69, 1)',
            backgroundColor: 'rgba(40, 167, 69, 0.2)',
            tension: 0.3,
            fill: true
        });

        // 2. Prediction Dataset (Dotted line)
        if (predictedLevel !== null) {
            const predictionDataset = Array(history.water_level.length).fill(null);
            // The prediction line starts at the last historical point and goes to the forecast point
            predictionDataset.push(history.water_level[history.water_level.length - 1], predictedLevel);

            waterChartInstance.data.datasets.push({
                label: 'Predicted Level',
                data: predictionDataset,
                borderColor: 'rgba(255, 99, 132, 1)',
                backgroundColor: 'transparent',
                borderDash: [5, 5],
                pointRadius: [0, 0, 5],
                pointBackgroundColor: 'rgba(255, 99, 132, 1)',
                tension: 0.3,
                fill: false
            });
        }
        
        waterChartInstance.update();
    }
}

// --- Full Page Chart Functions (Unified Fetch) ---

async function initializeFullRainChart(data) {
    if (fullRainChartInstance) fullRainChartInstance.destroy();

    const history = await fetchData(HISTORY_ENDPOINTS.rainfall);
    if (!history) return;
    
    const labels = history.labels;
    const rainfallData = history.data;
    const unit = history.unit || 'mm';

    fullRainChartInstance = createChart(
        'fullRainChart', 'bar', `Daily Rainfall (${unit})`, rainfallData, labels,
        'rgba(0, 123, 255, 0.8)', 'rgba(0, 123, 255, 0.5)', unit
    );
    
    // Update data cards
    const latestRain = rainfallData.length > 0 ? rainfallData[rainfallData.length - 1] : data.current.rainfall_mm;
    document.getElementById("currentRainIntensity").innerText = (latestRain / 6).toFixed(1); // Mock intensity based on latest reading
    document.getElementById("totalRain24h").innerText = latestRain.toFixed(1);
}

async function initializeFullWaterChart(data) {
    if (fullWaterChartInstance) fullWaterChartInstance.destroy();

    const history = await fetchData(HISTORY_ENDPOINTS.waterlevel);
    if (!history) return;

    const labels = history.labels;
    const waterData = history.data;
    const unit = history.unit || 'cm';

    fullWaterChartInstance = createChart(
        'fullWaterChart', 'line', `Daily Water Level (${unit})`, waterData, labels,
        'rgba(40, 167, 69, 1)', 'rgba(40, 167, 69, 0.4)', unit
    );

    // Update data cards
    const avgWaterLevel = waterData.length > 0 ? (waterData.reduce((a, b) => a + b, 0) / waterData.length).toFixed(1) : data.current.water_level_cm.toFixed(1);
    document.getElementById("avgWaterLevel").innerText = avgWaterLevel;
}

async function initializeFullTempChart(data) {
    if (fullTempChartInstance) fullTempChartInstance.destroy();

    const history = await fetchData(HISTORY_ENDPOINTS.temperature);
    if (!history) return;

    const labels = history.labels;
    const ambientData = history.data;
    const unit = history.unit || '°C';
    
    // Mock water temp data for the second dataset (since ThingSpeak only provides one field per channel entry)
    const waterData = ambientData.map(t => t - 2);

    fullTempChartInstance = createChart(
        'fullTempChart', 'line', `Ambient Temperature (${unit})`, ambientData, labels,
        'rgba(255, 99, 132, 1)', 'rgba(255, 99, 132, 0.4)', unit
    );
    
    // Add water temperature dataset
    fullTempChartInstance.data.datasets.push({
        label: `Water Temperature (${unit})`,
        data: waterData,
        borderColor: 'rgba(54, 162, 235, 1)',
        backgroundColor: 'rgba(54, 162, 235, 0.4)',
        tension: 0.3,
        fill: false
    });
    fullTempChartInstance.update();

    // Update data cards
    const latestTemp = ambientData.length > 0 ? ambientData[ambientData.length - 1] : data.current.temperature_c;
    document.getElementById("ambientTemp").innerText = latestTemp.toFixed(1);
    document.getElementById("waterTemp").innerText = (latestTemp - 2).toFixed(1);
}

async function initializeFullFlowChart(data) {
    if (fullFlowChartInstance) fullFlowChartInstance.destroy();

    const history = await fetchData(HISTORY_ENDPOINTS.flowrate);
    if (!history) return;

    const labels = history.labels;
    const flowData = history.data;
    const unit = history.unit || 'L/min';

    fullFlowChartInstance = createChart(
        'fullFlowChart', 'line', `Flow Rate (${unit})`, flowData, labels,
        'rgba(54, 162, 235, 1)', 'rgba(54, 162, 235, 0.4)', unit // Blue color for flow
    );

    // Update data cards
    const latestFlow = flowData.length > 0 ? flowData[flowData.length - 1] : data.current.flow_rate_mlpm;
    document.getElementById("currentFlowRate").innerText = latestFlow.toFixed(1);
    
    const avgFlowRate = flowData.length > 0 ? (flowData.reduce((a, b) => a + b, 0) / flowData.length).toFixed(1) : data.current.flow_rate_mlpm.toFixed(1);
    document.getElementById("avgFlowRate").innerText = avgFlowRate;
}

// --- Map Functions ---

let mapInstance;

function initializeMap() {
    // Coordinates centered near a typical Indian location (e.g., Agartala region)
    const initialCoords = [23.8314, 91.2868]; 
    
    // Check if map is already initialized
    if (mapInstance) {
        mapInstance.remove();
    }

    mapInstance = L.map('map').setView(initialCoords, 10);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(mapInstance);

    // Example Markers (replace with actual sensor locations)
    const markers = [
        { coords: [23.85, 91.30], name: "Zone 1", risk: "High" },
        { coords: [23.80, 91.25], name: "Zone 2", risk: "Low" },
        { coords: [23.88, 91.20], name: "Zone 3", risk: "Moderate" }
    ];

    markers.forEach(m => {
        let color = 'green';
        if (m.risk === 'Moderate') color = 'orange';
        if (m.risk === 'High') color = 'red';

        const customIcon = L.divIcon({
            className: `map-marker marker-${m.risk.toLowerCase()}`,
            html: `<div style="background-color: ${color}; width: 15px; height: 15px; border-radius: 50%; border: 2px solid white;"></div>`,
            iconSize: [20, 20],
            iconAnchor: [10, 10]
        });

        L.marker(m.coords, { icon: customIcon })
            .addTo(mapInstance)
            .bindPopup(`${m.name}: Flood Risk ${m.risk}`);
    });
}
// --- Routing and Page Rendering Logic ---

const ROUTES = {
    'dashboard': {
        templateId: 'dashboard-template',
        init: (data) => {
            renderDataCards(data.current);
            renderAlerts(data.alerts);
            initializeDashboardCharts(data.history, data.current);
            initializeMap(); // Re-initialize map when returning to dashboard
            // Re-attach location check listener
            document.getElementById('checkLocationBtn').addEventListener('click', checkLocationSafety);
        },
        update: (data) => {
            // Only update dynamic elements
            renderDataCards(data.current);
            renderAlerts(data.alerts);
            updateDashboardCharts(data.history, data.current);
            // Map markers are static in this implementation, so no update needed.
        }
    },
    'rainfall': { templateId: 'rainfall-template', init: (data) => {
        initializeFullRainChart(data);
    }},
    'waterlevel': { templateId: 'waterlevel-template', init: (data) => {
        initializeFullWaterChart(data);
    }},
    'temperature': { templateId: 'temperature-template', init: (data) => {
        initializeFullTempChart(data);
    }},
    'flowrate': { templateId: 'flowrate-template', init: (data) => { // New Route
        initializeFullFlowChart(data);
    }}
};

// Function to update data on the currently rendered page without re-rendering the structure
function updateCurrentPageData(page, data) {
    const route = ROUTES[page];
    if (route && route.update) {
        route.update(data);
    } else if (page === 'dashboard') {
        // Default update logic for dashboard if no specific update function exists
        renderDataCards(data.current);
        renderAlerts(data.alerts);
        updateDashboardCharts(data.history, data.current);
    }
}

function renderPage(page, data) {
    const appContent = document.getElementById('app-content');
    const route = ROUTES[page];

    if (!route) {
        appContent.innerHTML = '<h1>404 Page Not Found</h1>';
        return;
    }

    // 1. Load Template
    const template = document.getElementById(route.templateId);
    if (template) {
        // Clear existing content and append template clone
        appContent.innerHTML = '';
        appContent.appendChild(template.content.cloneNode(true));
    }

    // 2. Run Initialization/Data Binding
    // NOTE: We only run init when the page structure is first rendered.
    if (data) {
        route.init(data);
    }

    // 3. Update Navigation Links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('data-page') === page) {
            link.classList.add('active');
        }
    });
}

function handleRouting(data) {
    const hash = window.location.hash.substring(1) || 'dashboard';
    const page = hash.split('/')[0]; // Use only the main page name

    if (page !== currentPage) {
        // Only re-render the entire page structure if the route changes
        renderPage(page, data);
        currentPage = page;
    } else if (data) {
        // Otherwise, just update the data on the current page
        updateCurrentPageData(page, data);
    }
}

// --- End Routing and Page Rendering Logic ---

function pointLocationOnMap(locationName, coords) {
    if (!mapInstance) return;

    // 1. Remove existing search marker
    if (searchMarker) {
        mapInstance.removeLayer(searchMarker);
    }

    // 2. Center map on the new location
    mapInstance.setView(coords, 13); // Zoom in slightly

    // 3. Add a new marker
    searchMarker = L.marker(coords).addTo(mapInstance)
        .bindPopup(`Searched Location: ${locationName}`).openPopup();
}

// --- Location Safety Check Logic ---

// Mock function to check location safety based on current mock alerts
function checkLocationSafety() {
    const locationInput = document.getElementById('locationInput').value.trim();
    const resultElement = document.getElementById('locationSafetyResult');
    resultElement.className = 'safety-result'; // Reset classes
    resultElement.innerText = '';

    if (!locationInput) {
        resultElement.innerText = 'Please enter a location.';
        resultElement.classList.add('safety-unknown');
        return;
    }

    // New: Geocode the location input
    const normalizedInput = locationInput.toLowerCase();
    const coords = MOCK_LOCATIONS[normalizedInput];

    if (!coords) {
        resultElement.innerText = `Location "${locationInput}" not found in database.`;
        resultElement.classList.add('safety-unknown');
        // Optionally, clear the search marker if location is not found
        if (searchMarker) {
            mapInstance.removeLayer(searchMarker);
            searchMarker = null;
        }
        return;
    }

    // Point location on map
    pointLocationOnMap(locationInput, coords);

    // Check if the entered location matches any active alert zone
    if (!currentData) return; // Safety check
    const activeAlerts = currentData.alerts;
    const matchingAlert = activeAlerts.find(alert =>
        alert.zone.toLowerCase() === normalizedInput
    );

    if (matchingAlert) {
        // Location is unsafe/at risk
        resultElement.innerText = `ALERT: ${locationInput} has a ${matchingAlert.risk} flood risk! Message: ${matchingAlert.message}`;
        if (matchingAlert.risk === 'High') {
            resultElement.classList.add('safety-unsafe');
        } else {
            resultElement.classList.add('safety-unknown');
        }
    } else {
        // Assume safe if no active alert matches the zone
        resultElement.innerText = `${locationInput} is currently safe.`;
        resultElement.classList.add('safety-safe');
    }
}

// --- Main Fetch Logic (Unified Fetch) ---

async function fetchData(url) {
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error(`Error fetching data from ${url}:`, error);
        return null;
    }
}

async function fetchAndRouteData() {
    updateLastUpdateTime();

    // Fetch dashboard data (current state, history, alerts)
    const dashboardData = await fetchData(API_ENDPOINT);
    
    if (dashboardData) {
        console.log("Dashboard data fetched successfully from backend.");

        // Store data globally and pass to the router to render the current page
        currentData = dashboardData;
        handleRouting(dashboardData);
    }
}

// --- Polling Management ---

function startAutoRefresh() {
    // Clear any existing interval to prevent multiple simultaneous polls
    if (refreshIntervalId) {
        clearInterval(refreshIntervalId);
    }
    refreshIntervalId = setInterval(fetchAndRouteData, REFRESH_INTERVAL_MS);
}

// --- Initialization ---

document.addEventListener('DOMContentLoaded', () => {
    // 1. Initialize Dark Mode
    initializeDarkMode();

    // 2. Set up routing listeners
    window.addEventListener('hashchange', () => {
        // Fetch data on hash change, but do not restart the interval here.
        fetchAndRouteData();
    });
    
    // 3. Fetch initial data and render the first page
    fetchAndRouteData();

    // 4. Set up auto-refresh
    startAutoRefresh();
});
