// ============================================================================
// FILE 3: FRONTEND INTEGRATION - api-integration.js
// Add this to your HTML file before </body>
// ============================================================================

// API Configuration
const API_BASE_URL = 'http://localhost:5000/api';

let isBackendAvailable = false;

// ============================================================================
// Initialize Backend Connection
// ============================================================================

async function initializeBackend() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            const data = await response.json();
            isBackendAvailable = true;
            console.log('✓ Backend connected. Activities loaded:', data.activities_loaded);
            return true;
        }
    } catch (error) {
        console.warn('Backend not available:', error);
        isBackendAvailable = false;
    }
    return false;
}

// ============================================================================
// Main Recommendation Function
// ============================================================================

async function getHybridRecommendations(query, groupMembers = [], preferences = [], topK = 5) {
    /**
     * Call backend to get recommendations
     * 
     * Flow:
     * 1. BM25 keyword search
     * 2. Dense semantic search
     * 3. Reciprocal Rank Fusion (merge)
     * 4. Score by group member details (age, preferences)
     * 5. Return ranked results
     */
    
    if (!isBackendAvailable) {
        console.warn('Backend not available');
        return { status: 'error', recommendations: [] };
    }
    
    try {
        const payload = {
            query: query,
            group_members: groupMembers.map(m => ({ age: m.age })),
            preferences: preferences,
            top_k: topK
        };
        
        console.log('Sending recommendation request:', payload);
        
        const response = await fetch(`${API_BASE_URL}/recommend`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            console.log('✓ Got', result.count, 'recommendations');
            return result;
        } else {
            console.error('Recommendation failed:', result);
            return { status: 'error', recommendations: [] };
        }
    } catch (error) {
        console.error('Error fetching recommendations:', error);
        return { status: 'error', recommendations: [] };
    }
}

// ============================================================================
// Search Integration (Hook into your existing search bar)
// ============================================================================

async function performSearch(query) {
    /**
     * Called when user searches
     * Sends query + group context to backend
     */
    
    console.log('Searching for:', query);
    
    if (!isBackendAvailable) {
        console.warn('Backend unavailable, using client-side search');
        return;
    }
    
    // Get group members from UI
    const groupMembers = window.groupMembers || [];
    
    // Get preferences/filters from UI
    const preferences = [];
    if (document.querySelectorAll('input[name="preferences"]:checked').length > 0) {
        document.querySelectorAll('input[name="preferences"]:checked').forEach(cb => {
            preferences.push(cb.value);
        });
    }
    
    // Call backend
    const result = await getHybridRecommendations(
        query,
        groupMembers,
        preferences,
        10
    );
    
    if (result.status === 'success') {
        // Display results (adapt to your HTML structure)
        displayRecommendations(result.recommendations);
    }
}

// ============================================================================
// Display Results (Customize based on your HTML)
// ============================================================================

function displayRecommendations(recommendations) {
    /**
     * Shows recommendations in your UI
     * Adapt this to match your HTML elements
     */
    
    console.log('Displaying', recommendations.length, 'recommendations');
    
    // Example: Update a results container
    const resultsContainer = document.getElementById('results') || 
                             document.getElementById('activities-container') ||
                             document.querySelector('.activities-grid');
    
    if (!resultsContainer) {
        console.warn('Results container not found');
        return;
    }
    
    // Clear previous results
    resultsContainer.innerHTML = '';
    
    // Display each recommendation
    recommendations.forEach((rec, index) => {
        const card = document.createElement('div');
        card.className = 'activity-card';
        card.innerHTML = `
            <div class="activity-rank">Rank: ${rec.rank}</div>
            <div class="activity-score">Score: ${(rec.recommendation_score * 100).toFixed(1)}%</div>
            <h3>${rec.title || rec.name || 'Activity'}</h3>
            <p>${rec.description || 'No description'}</p>
            <div class="activity-meta">
                ${rec.age_min ? `<span>Ages ${rec.age_min}-${rec.age_max}</span>` : ''}
                ${rec.duration_mins ? `<span>${rec.duration_mins} mins</span>` : ''}
                ${rec.cost ? `<span>Cost: ${rec.cost}</span>` : ''}
            </div>
            ${rec.tags ? `<div class="activity-tags">${rec.tags}</div>` : ''}
        `;
        resultsContainer.appendChild(card);
    });
}

// ============================================================================
// Get Personalized Recommendations (for group members)
// ============================================================================

async function getPersonalizedRecommendations() {
    /**
     * Called when user clicks "Get Recommendations" button
     * Uses group member ages & interests for personalization
     */
    
    const groupMembers = window.groupMembers || [];
    
    if (groupMembers.length === 0) {
        alert('Please add group members first');
        return;
    }
    
    // Extract group profile
    const ages = groupMembers.map(m => m.age || 10);
    const minAge = Math.min(...ages);
    const maxAge = Math.max(...ages);
    const allInterests = [...new Set(groupMembers.flatMap(m => m.interests || []))];
    
    // Build query from interests
    const query = allInterests.slice(0, 3).join(' ') || 'fun family activities';
    
    console.log(`Group: ages ${minAge}-${maxAge}, interests: ${allInterests.join(', ')}`);
    
    // Get recommendations
    const result = await getHybridRecommendations(
        query,
        groupMembers,
        allInterests,
        10
    );
    
    if (result.status === 'success') {
        displayRecommendations(result.recommendations);
        alert(`Found ${result.count} recommendations for your group!`);
    }
}

// ============================================================================
// Initialize on Page Load
// ============================================================================

document.addEventListener('DOMContentLoaded', async function() {
    console.log('Initializing...');
    
    // Connect to backend
    await initializeBackend();
    
    // Hook up search button (adjust selector to match your HTML)
    const searchBtn = document.querySelector('button[onclick*="search"]') ||
                      document.getElementById('searchBtn') ||
                      document.querySelector('.search-button');
    
    if (searchBtn) {
        searchBtn.addEventListener('click', () => {
            const query = document.querySelector('input[name="query"]').value ||
                          document.getElementById('searchInput').value;
            performSearch(query);
        });
    }
    
    // Hook up recommendations button
    const recBtn = document.querySelector('button[onclick*="recommendation"]') ||
                   document.getElementById('getRecommendationsBtn') ||
                   document.querySelector('.recommendations-button');
    
    if (recBtn) {
        recBtn.addEventListener('click', getPersonalizedRecommendations);
    }
    
    // Hook up search input (Enter key)
    const searchInput = document.querySelector('input[name="query"]') ||
                        document.getElementById('searchInput');
    
    if (searchInput) {
        searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                performSearch(e.target.value);
            }
        });
    }
    
    console.log('✓ Frontend initialization complete');
});

// ============================================================================
// Export functions globally for onclick handlers
// ============================================================================

window.performSearch = performSearch;
window.getPersonalizedRecommendations = getPersonalizedRecommendations;
window.getHybridRecommendations = getHybridRecommendations;
