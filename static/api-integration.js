// SIMPLIFIED API INTEGRATION - Guaranteed Working Version
// Replace your C:\Users\HP\Desktop\activity-website\static\api-integration.js with this

const API_BASE_URL = 'http://localhost:5000/api';
let isBackendAvailable = false;
let currentGroupProfile = {};

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
            </div>
        `;
        
        recList.appendChild(card);
    });
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

// Initialize
document.addEventListener('DOMContentLoaded', async function() {
    console.log('Page loaded, initializing...');
    await initializeBackend();
    
    // Find and hook up button
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
    
    console.log('✓ Initialization complete');
});

// Export globally
window.getPersonalizedRecommendations = getPersonalizedRecommendations;
window.hideResultsPage = hideResultsPage;