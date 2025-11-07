// --- Configuration ---
// Replace this with your actual Firebase API URL
const API_ENDPOINT = "http://127.0.0.1:5000/data";
// const API_ENDPOINT = "./data.json"; // No longer fetching static data
const REFRESH_INTERVAL_MS = 5000; // 5 seconds (Adjusted frequency to balance responsiveness and performance)

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
    // Note: Chart colors might need explicit update if they don't inherit CSS variables well.
    // For now, we rely on CSS, but this is where chart updates would go if needed.
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

function renderDataCards(data) {
    const dataElements = [
        { id: "rain", value: data.rainfall_mm.toFixed(1) },
        { id: "water", value: data.water_level_cm.toFixed(1) },
        { id: "temp", value: data.temperature_c.toFixed(1) },
        { id: "risk", value: data.ai_risk, isRisk: true }
    ];

    dataElements.forEach(item => {
        const element = document.getElementById(item.id);
        if (element.innerText !== item.value) {
            // Apply animation class if data has changed
            element.classList.remove('animate-update');
            // Force reflow/repaint to restart animation
            void element.offsetWidth;
            element.classList.add('animate-update');
        }
        element.innerText = item.value;

        if (item.isRisk) {
            element.className = `data-value risk-level risk-${item.value.toLowerCase()}`;
        }
    });

    document.getElementById("ai-message").innerText = data.ai_message;
    document.getElementById("ai-recommendation").innerText = data.ai_recommendation;

    updateSystemStatus(data.ai_risk);
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

// --- Chart Functions ---

const CHART_OPTIONS = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
        y: {
            beginAtZero: true
        }
    },
    plugins: {
        legend: {
            display: false
        }
    }
};

function createChart(elementId, type, label, data, labels, borderColor, backgroundColor) {
    const ctx = document.getElementById(elementId);
    if (!ctx) return null; // Return null if element doesn't exist (page not loaded)

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
        options: CHART_OPTIONS
    });
}

function initializeDashboardCharts(history) {
    // Destroy existing instances if they exist (important for dynamic loading)
    if (rainChartInstance) rainChartInstance.destroy();
    if (waterChartInstance) waterChartInstance.destroy();

    // Rainfall Chart (Dashboard)
    rainChartInstance = createChart(
        'rainChart', 'line', 'Rainfall (mm)', history.rainfall, history.labels,
        'rgba(0, 123, 255, 1)', 'rgba(0, 123, 255, 0.2)'
    );

    // Water Level Chart (Dashboard)
    waterChartInstance = createChart(
        'waterChart', 'line', 'Water Level (cm)', history.water_level, history.labels,
        'rgba(40, 167, 69, 1)', 'rgba(40, 167, 69, 0.2)'
    );
}

function updateDashboardCharts(history) {
    if (rainChartInstance) {
        rainChartInstance.data.labels = history.labels;
        rainChartInstance.data.datasets[0].data = history.rainfall;
        rainChartInstance.update();
    }

    if (waterChartInstance) {
        waterChartInstance.data.labels = history.labels;
        waterChartInstance.data.datasets[0].data = history.water_level;
        waterChartInstance.update();
    }
}

// --- Full Page Chart Functions (Simulated Dynamic Data) ---

function initializeFullRainChart(data) {
    if (fullRainChartInstance) fullRainChartInstance.destroy();

    // Simulate 7 days of data
    const labels = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'];
    const rainfallData = [10, 5, 20, 15, 8, 12, data.current.rainfall_mm];

    fullRainChartInstance = createChart(
        'fullRainChart', 'bar', 'Daily Rainfall (mm)', rainfallData, labels,
        'rgba(0, 123, 255, 0.8)', 'rgba(0, 123, 255, 0.5)'
    );
    
    // Update data cards
    document.getElementById("currentRainIntensity").innerText = (data.current.rainfall_mm / 6).toFixed(1); // Mock intensity
    document.getElementById("totalRain24h").innerText = data.current.rainfall_mm.toFixed(1);
}

function initializeFullWaterChart(data) {
    if (fullWaterChartInstance) fullWaterChartInstance.destroy();

    // Simulate 7 days of data
    const labels = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'];
    const waterData = [35, 40, 45, 50, 48, 46, data.current.water_level_cm];

    fullWaterChartInstance = createChart(
        'fullWaterChart', 'line', 'Daily Water Level (cm)', waterData, labels,
        'rgba(40, 167, 69, 1)', 'rgba(40, 167, 69, 0.4)'
    );

    // Update data cards
    document.getElementById("avgWaterLevel").innerText = (waterData.reduce((a, b) => a + b, 0) / waterData.length).toFixed(1);
}

function initializeFullTempChart(data) {
    if (fullTempChartInstance) fullTempChartInstance.destroy();

    // Simulate 7 days of data
    const labels = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'];
    const ambientData = [25, 26, 28, 27, 29, 30, data.current.temperature_c];
    const waterData = [22, 23, 24, 23, 25, 26, data.current.temperature_c - 2]; // Mock water temp slightly lower

    fullTempChartInstance = createChart(
        'fullTempChart', 'line', 'Ambient Temperature (°C)', ambientData, labels,
        'rgba(255, 99, 132, 1)', 'rgba(255, 99, 132, 0.4)'
    );
    
    // Add water temperature dataset
    fullTempChartInstance.data.datasets.push({
        label: 'Water Temperature (°C)',
        data: waterData,
        borderColor: 'rgba(54, 162, 235, 1)',
        backgroundColor: 'rgba(54, 162, 235, 0.4)',
        tension: 0.3,
        fill: false
    });
    fullTempChartInstance.update();

    // Update data cards
    document.getElementById("ambientTemp").innerText = data.current.temperature_c.toFixed(1);
    document.getElementById("waterTemp").innerText = (data.current.temperature_c - 2).toFixed(1);
}

// --- Map Functions ---

let mapInstance;

function initializeMap() {
    // Coordinates centered near a typical Indian location (e.g., Agartala region)
    const initialCoords = [23.8314, 91.2868]; 
    
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
            initializeDashboardCharts(data.history);
            initializeMap(); // Re-initialize map when returning to dashboard
            // Re-attach location check listener
            document.getElementById('checkLocationBtn').addEventListener('click', checkLocationSafety);
        },
        update: (data) => {
            // Only update dynamic elements
            renderDataCards(data.current);
            renderAlerts(data.alerts);
            updateDashboardCharts(data.history);
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
        updateDashboardCharts(data.history);
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

// --- Main Fetch Logic ---

async function fetchData() {
    updateLastUpdateTime();

    try {
        const response = await fetch(API_ENDPOINT);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        
        console.log("Data fetched successfully from backend.");

        // Store data globally and pass to the router to render the current page
        currentData = data;
        handleRouting(data);

    } catch (error) {
        console.error("Error fetching data from backend:", error);
        // Fallback or display error message on frontend if needed
        // For now, we just stop updating if the fetch fails.
    }
}

// --- Polling Management ---

function startAutoRefresh() {
    // Clear any existing interval to prevent multiple simultaneous polls
    if (refreshIntervalId) {
        clearInterval(refreshIntervalId);
    }
    refreshIntervalId = setInterval(fetchData, REFRESH_INTERVAL_MS);
}

// --- Initialization ---

document.addEventListener('DOMContentLoaded', () => {
    // 1. Initialize Dark Mode
    initializeDarkMode();

    // 2. Set up routing listeners
    window.addEventListener('hashchange', () => {
        // Fetch data on hash change, but do not restart the interval here.
        fetchData();
    });
    
    // 3. Fetch initial data and render the first page
    fetchData();

    // 4. Set up auto-refresh
    startAutoRefresh();
    
    // Note: Map initialization and location check listener are now handled inside the 'dashboard' route init function.
});