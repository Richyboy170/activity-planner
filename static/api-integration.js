// SIMPLIFIED API INTEGRATION - Guaranteed Working Version
// Replace your C:\Users\HP\Desktop\activity-website\static\api-integration.js with this

const API_BASE_URL = 'http://localhost:5000/api';
let isBackendAvailable = false;
let currentGroupProfile = {};
// Note: currentActivities is declared in the main HTML file
// Note: savedActivities uses localStorage

// Initialize
async function initializeBackend() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            isBackendAvailable = true;
            console.log('✓ Backend connected');
            return true;
        }
    } catch (error) {
        console.error('Backend connection error:', error);
        isBackendAvailable = false;
    }
    return false;
}

// Get recommendations
async function getPersonalizedRecommendations() {
    console.log('=== GET RECOMMENDATIONS CLICKED ===');
    
    // Ensure groupMembers is initialized
    if (!window.groupMembers) {
        window.groupMembers = [];
    }
    
    const groupMembers = window.groupMembers;
    console.log('Group members:', groupMembers);
    console.log('Group members count:', groupMembers.length);
    
    if (groupMembers.length === 0) {
        alert('Please add at least one group member before getting recommendations.\n\nUse the "Add Member" form or click one of the "Quick Add Templates" buttons.');
        return;
    }
    
    if (!isBackendAvailable) {
        console.error('Backend not available!');
        alert('Backend not available. Make sure Flask is running.');
        return;
    }
    
    // Extract group info
    const ages = groupMembers.map(m => m.age || 10);
    const minAge = Math.min(...ages);
    const maxAge = Math.max(...ages);
    const avgAbility = Math.round((groupMembers.reduce((sum, m) => sum + (m.overall_ability || 5), 0) / groupMembers.length) * 10) / 10;
    const allInterests = [...new Set(groupMembers.flatMap(m => m.interests || []))];
    
    currentGroupProfile = {
        members: groupMembers,
        totalMembers: groupMembers.length,
        ageRange: `${minAge}-${maxAge}`,
        avgAbility: avgAbility,
        interests: allInterests
    };
    
    console.log('Group profile:', currentGroupProfile);
    
    const query = allInterests.slice(0, 3).join(' ') || 'fun family activities';
    console.log('Query:', query);
    
    try {
        const payload = {
            query: query,
            group_members: groupMembers.map(m => ({
                age: m.age,
                name: m.name,
                overall_ability: m.overall_ability,
                interests: m.interests || [],
                special_needs: m.special_needs || [],
                allergies: m.allergies || ''
            })),
            preferences: allInterests,
            top_k: 15
        };
        
        console.log('Sending payload:', payload);
        
        const response = await fetch(`${API_BASE_URL}/recommend`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        console.log('Response status:', response.status);
        
        const result = await response.json();
        console.log('Response data:', result);
        
        if (result.status === 'success' && result.recommendations) {
            console.log('✓ Got recommendations:', result.recommendations.length);
            showResultsPage(result.recommendations);
        } else {
            console.error('No recommendations in response');
            alert('No recommendations found. Check console for errors.');
        }
    } catch (error) {
        console.error('ERROR fetching recommendations:', error);
        alert('Error: ' + error.message);
    }
}

// Show Results Page
function showResultsPage(recommendations) {
    console.log('=== SHOWING RESULTS PAGE ===');
    
    // Hide main content
    const mainPage = document.getElementById('main-page');
    if (mainPage) {
        console.log('Hiding main page');
        mainPage.style.display = 'none';
    } else {
        console.warn('Main page element not found - hiding body content instead');
        // Hide all direct children of body except results
        Array.from(document.body.children).forEach(child => {
            if (child.id !== 'results-page') {
                child.style.display = 'none';
            }
        });
    }    
    // Create or show results page
    let resultsPage = document.getElementById('results-page');
    if (!resultsPage) {
        console.log('Creating results page container...');
        createResultsPageContainer();
        resultsPage = document.getElementById('results-page');
    }
    
    resultsPage.style.display = 'block';
    populateResultsPage(recommendations);
    
    console.log('Results page displayed');
}

// Create Results Container
function createResultsPageContainer() {
    const container = document.createElement('div');
    container.id = 'results-page';
    container.innerHTML = `
        <div class="results-page-wrapper">
            <div class="results-header">
                <button id="back-to-home" class="btn-back">← Back to Home</button>
                <h1>Personalized Activity Recommendations</h1>
            </div>
            <div class="results-container">
                <div class="group-profile-section">
                    <h2>Your Group Profile</h2>
                    <div id="group-profile"></div>
                </div>
                <div class="recommendations-section">
                    <h2>Top Ranked Activities</h2>
                    <div id="recommendations-list" class="recommendations-list"></div>
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(container);
    document.getElementById('back-to-home').addEventListener('click', hideResultsPage);
}

// Populate Results
function populateResultsPage(recommendations) {
    console.log('Populating results page with', recommendations.length, 'recommendations');

    // Show group profile
    const profileDiv = document.getElementById('group-profile');
    const profile = currentGroupProfile;

    let membersList = '<div class="members-list">';
    profile.members.forEach((member, i) => {
        membersList += `
            <div class="member-badge">
                <strong>${member.name || 'Member ' + (i+1)}</strong><br>
                Age: ${member.age}<br>
                Ability: ${member.overall_ability || 5}/10
            </div>
        `;
    });
    membersList += '</div>';

    profileDiv.innerHTML = `
        <div class="profile-stats">
            <div class="stat">
                <strong>Total Members:</strong> ${profile.totalMembers}
            </div>
            <div class="stat">
                <strong>Age Range:</strong> ${profile.ageRange} years
            </div>
            <div class="stat">
                <strong>Avg Ability:</strong> ${profile.avgAbility}/10
            </div>
            <div class="stat">
                <strong>Interests:</strong> ${profile.interests.slice(0, 3).join(', ')}
            </div>
        </div>
        ${membersList}
    `;

    // Show recommendations
    const recList = document.getElementById('recommendations-list');
    recList.innerHTML = '';

    recommendations.forEach((rec, index) => {
        const card = document.createElement('div');
        card.className = 'recommendation-card';
        card.style.cursor = 'pointer';
        card.onclick = () => showActivityDetails(rec, index);

        const scorePercent = Math.round((rec.recommendation_score || 0) * 100);

        card.innerHTML = `
            <div class="card-header">
                <div class="rank-badge">#${index + 1}</div>
                <div class="score-badge" style="background-color: ${scorePercent >= 80 ? '#2ecc71' : scorePercent >= 60 ? '#f39c12' : '#e74c3c'};">
                    ${scorePercent}% Match
                </div>
            </div>
            <div class="card-content">
                <h3>${rec.title || 'Activity'}</h3>
                <p>${rec.description || 'No description'}</p>
                <div class="activity-details">
                    ${rec.duration_mins ? 'Duration: ' + rec.duration_mins + ' mins<br>' : ''}
                    ${rec.cost ? 'Cost: ' + rec.cost + '<br>' : ''}
                    ${rec.location ? 'Location: ' + rec.location + '<br>' : ''}
                </div>
                <button class="btn btn--primary btn--sm" style="margin-top: 12px;" onclick="event.stopPropagation(); addToMyPlan(${index})">Add to My Plan</button>
            </div>
        `;

        recList.appendChild(card);
    });

    // Update currentActivities in global scope (defined in HTML)
    if (typeof window.currentActivities !== 'undefined') {
        window.currentActivities = recommendations;
    }
}

// Hide Results & Show Home
function hideResultsPage() {
    console.log('=== HIDING RESULTS PAGE ===');
    
    const mainPage = document.getElementById('main-page');
    const resultsPage = document.getElementById('results-page');
    
    if (resultsPage) {
        resultsPage.style.display = 'none';
    }
    
    if (mainPage) {
        mainPage.style.display = 'block';
    } else {
        // Show all hidden children
        Array.from(document.body.children).forEach(child => {
            if (child.id !== 'results-page') {
                child.style.display = '';
            }
        });
    }
    
    window.scrollTo(0, 0);
}

// Show Activity Details
function showActivityDetails(activity, index) {
    console.log('Showing activity details:', activity);

    const detailsContent = document.getElementById('activityDetailsContent');
    const detailsTitle = document.getElementById('activityDetailsTitle');
    const breadcrumb = document.getElementById('activityDetailsBreadcrumb');

    if (detailsTitle) detailsTitle.textContent = activity.title || 'Activity Details';
    if (breadcrumb) breadcrumb.textContent = activity.title || 'Activity Details';

    // Parse materials and how_to_play if they're strings
    let materials = [];
    let howToPlay = [];

    try {
        materials = typeof activity.materials_needed === 'string'
            ? JSON.parse(activity.materials_needed.replace(/'/g, '"'))
            : (activity.materials_needed || []);
    } catch (e) {
        materials = activity.materials_needed || [];
    }

    try {
        howToPlay = typeof activity.how_to_play === 'string'
            ? JSON.parse(activity.how_to_play.replace(/'/g, '"'))
            : (activity.how_to_play || []);
    } catch (e) {
        howToPlay = activity.how_to_play || [];
    }

    const materialsHtml = Array.isArray(materials) && materials.length > 0
        ? `<div style="margin-bottom: 24px;">
            <h3 style="margin-bottom: 12px;">Materials Needed</h3>
            <ul style="list-style: disc; padding-left: 20px;">
                ${materials.map(m => `<li>${m}</li>`).join('')}
            </ul>
          </div>`
        : '';

    const howToPlayHtml = Array.isArray(howToPlay) && howToPlay.length > 0
        ? `<div style="margin-bottom: 24px;">
            <h3 style="margin-bottom: 12px;">How to Play</h3>
            <ol style="list-style: decimal; padding-left: 20px;">
                ${howToPlay.map(step => `<li style="margin-bottom: 8px;">${step}</li>`).join('')}
            </ol>
          </div>`
        : '';

    const scorePercent = activity.recommendation_score
        ? Math.round(activity.recommendation_score * 100)
        : null;

    detailsContent.innerHTML = `
        <div class="card" style="padding: 32px;">
            ${scorePercent ? `
                <div style="display: inline-block; background-color: ${scorePercent >= 80 ? '#2ecc71' : scorePercent >= 60 ? '#f39c12' : '#e74c3c'}; color: white; padding: 8px 16px; border-radius: 20px; margin-bottom: 16px; font-weight: bold;">
                    ${scorePercent}% Match
                </div>
            ` : ''}

            <p style="font-size: 18px; line-height: 1.6; margin-bottom: 24px;">
                ${activity.description || 'No description available'}
            </p>

            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; padding: 20px; background-color: #f8f9fa; border-radius: 8px;">
                ${activity.age_min || activity.age_max ? `
                    <div>
                        <strong>Age Range:</strong><br>
                        ${activity.age_min || '?'} - ${activity.age_max || '?'} years
                    </div>
                ` : ''}
                ${activity.duration_mins ? `
                    <div>
                        <strong>Duration:</strong><br>
                        ${activity.duration_mins} minutes
                    </div>
                ` : ''}
                ${activity.cost ? `
                    <div>
                        <strong>Cost:</strong><br>
                        ${activity.cost}
                    </div>
                ` : ''}
                ${activity.indoor_outdoor ? `
                    <div>
                        <strong>Location:</strong><br>
                        ${activity.indoor_outdoor}
                    </div>
                ` : ''}
                ${activity.season ? `
                    <div>
                        <strong>Season:</strong><br>
                        ${activity.season}
                    </div>
                ` : ''}
                ${activity.players ? `
                    <div>
                        <strong>Players:</strong><br>
                        ${activity.players}
                    </div>
                ` : ''}
            </div>

            ${activity.parent_caution && activity.parent_caution.toLowerCase() === 'yes' ? `
                <div style="background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 16px; margin-bottom: 24px; border-radius: 4px;">
                    <strong>⚠️ Parent Caution:</strong> This activity requires adult supervision.
                </div>
            ` : ''}

            ${activity.tags ? `
                <div style="margin-bottom: 24px;">
                    <strong>Tags:</strong><br>
                    ${activity.tags.split(',').map(tag =>
                        `<span style="display: inline-block; background-color: #e9ecef; padding: 4px 12px; border-radius: 12px; margin: 4px; font-size: 14px;">${tag.trim()}</span>`
                    ).join('')}
                </div>
            ` : ''}

            ${materialsHtml}
            ${howToPlayHtml}

            <div style="margin-top: 32px; display: flex; gap: 12px;">
                <button class="btn btn--primary" onclick="addToMyPlanFromDetails(${index})">
                    Add to My Plan
                </button>
                <button class="btn btn--outline" onclick="showView('search')">
                    ← Back to Results
                </button>
            </div>
        </div>
    `;

    // Hide results page and show activity details
    const resultsPage = document.getElementById('results-page');
    if (resultsPage) resultsPage.style.display = 'none';

    showView('activity-details');
}

// Add to My Plan
function addToMyPlan(index) {
    const activity = (typeof window.currentActivities !== 'undefined' ? window.currentActivities : [])[index];
    if (!activity) return;

    const savedActivities = JSON.parse(localStorage.getItem('savedActivities') || '[]');
    if (!savedActivities.find(a => a.title === activity.title)) {
        savedActivities.push(activity);
        localStorage.setItem('savedActivities', JSON.stringify(savedActivities));
        alert(`"${activity.title}" added to My Plan!`);
    } else {
        alert(`"${activity.title}" is already in My Plan.`);
    }
}

function addToMyPlanFromDetails(index) {
    addToMyPlan(index);
}

// Load Featured Activities for Home Page
async function loadFeaturedActivities() {
    if (!isBackendAvailable) return;

    try {
        const response = await fetch(`${API_BASE_URL}/activities?limit=12`);
        const data = await response.json();

        if (data.status === 'success') {
            displayActivities(data.activities, 'featuredActivities');
        }
    } catch (error) {
        console.error('Error loading featured activities:', error);
    }
}

// Search Activities
async function performSearch() {
    if (!isBackendAvailable) {
        alert('Backend not available. Please ensure Flask is running.');
        return;
    }

    const query = document.getElementById('searchInput')?.value ||
                  document.getElementById('mainSearchInput')?.value || '';
    const minAge = parseInt(document.getElementById('minAge')?.value || 2);
    const maxAge = parseInt(document.getElementById('maxAge')?.value || 18);
    const duration = document.getElementById('durationFilter')?.value || '';
    const cost = document.getElementById('costFilter')?.value || '';
    const season = document.getElementById('seasonFilter')?.value || '';
    const players = document.getElementById('playersFilter')?.value || '';

    // Get active indoor/outdoor filter
    const activeLocationBtn = document.querySelector('.toggle-btn.active[data-filter="indoor_outdoor"]');
    const indoorOutdoor = activeLocationBtn?.dataset.value || '';

    const filters = {
        min_age: minAge,
        max_age: maxAge
    };

    if (duration) filters.duration = parseInt(duration);
    if (cost) filters.cost = cost;
    if (indoorOutdoor) filters.indoor_outdoor = indoorOutdoor;
    if (season) filters.season = season;
    if (players) filters.players = players;

    try {
        const response = await fetch(`${API_BASE_URL}/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, filters, limit: 50 })
        });

        const data = await response.json();

        if (data.status === 'success') {
            if (typeof window.currentActivities !== 'undefined') {
                window.currentActivities = data.activities;
            }
            displayActivities(data.activities, 'activitiesGrid');
            if (typeof showView === 'function') {
                showView('search');
            }
        }
    } catch (error) {
        console.error('Error searching activities:', error);
        alert('Search failed: ' + error.message);
    }
}

// Display Activities in Grid
function displayActivities(activities, containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = '';

    activities.forEach((activity, index) => {
        const card = document.createElement('div');
        card.className = 'activity-card';
        card.style.cursor = 'pointer';
        card.onclick = () => {
            if (typeof window.currentActivities !== 'undefined') {
                window.currentActivities = activities;
            }
            showActivityDetails(activity, index);
        };

        card.innerHTML = `
            <div class="card-content">
                <h3 style="margin-bottom: 12px;">${activity.title || 'Activity'}</h3>
                <p style="font-size: 14px; color: #6c757d; margin-bottom: 12px;">
                    ${(activity.description || '').substring(0, 100)}${(activity.description || '').length > 100 ? '...' : ''}
                </p>
                <div style="font-size: 13px; color: #6c757d;">
                    ${activity.age_min && activity.age_max ? `Ages: ${activity.age_min}-${activity.age_max}<br>` : ''}
                    ${activity.duration_mins ? `Duration: ${activity.duration_mins} mins<br>` : ''}
                    ${activity.cost ? `Cost: ${activity.cost}` : ''}
                </div>
            </div>
        `;

        container.appendChild(card);
    });

    if (activities.length === 0) {
        container.innerHTML = '<p style="text-align: center; color: #6c757d; padding: 40px;">No activities found. Try adjusting your filters.</p>';
    }
}

// Load My Plan activities
function loadMyPlan() {
    const planContainer = document.getElementById('myPlanActivities');
    if (!planContainer) return;

    const savedActivities = JSON.parse(localStorage.getItem('savedActivities') || '[]');

    if (savedActivities.length === 0) {
        planContainer.innerHTML = '<p style="text-align: center; color: #6c757d; padding: 40px;">No activities saved yet. Add activities from search or recommendations!</p>';
        return;
    }

    displayActivities(savedActivities, 'myPlanActivities');
}

// Initialize
document.addEventListener('DOMContentLoaded', async function() {
    console.log('Page loaded, initializing...');
    await initializeBackend();

    // Load featured activities
    loadFeaturedActivities();

    // Find and hook up recommendation button
    let button = document.getElementById('getRecommendationsBtn');
    if (!button) button = document.querySelector('[onclick*="recommendation"]');
    if (!button) button = document.querySelector('button:contains("Recommendations")');
    if (!button) {
        console.warn('Recommendations button not found, searching all buttons...');
        const allButtons = document.querySelectorAll('button');
        console.log('Found buttons:', allButtons.length);
        allButtons.forEach((btn, i) => {
            console.log(`Button ${i}:`, btn.textContent.substring(0, 50));
        });
    }

    if (button) {
        console.log('Found recommendations button, attaching listener');
        button.addEventListener('click', getPersonalizedRecommendations);
    } else {
        console.warn('Could not find recommendations button - you may need to click manually');
        window.getPersonalizedRecommendations = getPersonalizedRecommendations;
    }

    // Hook up search functionality
    const searchBtn = document.getElementById('searchBtn');
    if (searchBtn) searchBtn.addEventListener('click', performSearch);

    const mainSearchInput = document.getElementById('mainSearchInput');
    if (mainSearchInput) {
        mainSearchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                // Copy value to search page input
                const searchInput = document.getElementById('searchInput');
                if (searchInput) searchInput.value = mainSearchInput.value;
                performSearch();
            }
        });
    }

    const searchInput = document.getElementById('searchInput');
    if (searchInput) {
        searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') performSearch();
        });
    }

    // Hook up filter sliders
    const minAgeSlider = document.getElementById('minAge');
    const maxAgeSlider = document.getElementById('maxAge');
    const minAgeValue = document.getElementById('minAgeValue');
    const maxAgeValue = document.getElementById('maxAgeValue');

    if (minAgeSlider && minAgeValue) {
        minAgeSlider.addEventListener('input', (e) => {
            minAgeValue.textContent = e.target.value;
        });
    }

    if (maxAgeSlider && maxAgeValue) {
        maxAgeSlider.addEventListener('input', (e) => {
            maxAgeValue.textContent = e.target.value;
        });
    }

    // Hook up toggle buttons
    document.querySelectorAll('.toggle-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const filter = this.dataset.filter;
            document.querySelectorAll(`[data-filter="${filter}"]`).forEach(b => {
                b.classList.remove('active');
            });
            this.classList.add('active');
        });
    });

    console.log('✓ Initialization complete');
});

// Export globally
window.getPersonalizedRecommendations = getPersonalizedRecommendations;
window.hideResultsPage = hideResultsPage;
window.showActivityDetails = showActivityDetails;
window.addToMyPlan = addToMyPlan;
window.addToMyPlanFromDetails = addToMyPlanFromDetails;
window.performSearch = performSearch;
window.loadFeaturedActivities = loadFeaturedActivities;
window.loadMyPlan = loadMyPlan;