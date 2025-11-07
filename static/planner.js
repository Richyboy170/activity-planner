// Activity Planner - Drag and Drop Calendar functionality
// Google Calendar-style planning interface

// Store planned activities with timestamps
let plannedActivities = JSON.parse(localStorage.getItem('plannedActivities') || '[]');
let currentSearchResults = []; // Store current search results for easy access

// Save planned activities to localStorage
function savePlannedActivities() {
    localStorage.setItem('plannedActivities', JSON.stringify(plannedActivities));
}

// Add activity to plan
function addToPlan(activityData) {
    console.log('Adding to plan:', activityData);

    // If activityData is just a title/ID, find it in current search results
    let activity;
    if (typeof activityData === 'string' || typeof activityData === 'number') {
        activity = currentSearchResults.find(a =>
            a.title === activityData || a.id === activityData
        );
        if (!activity) {
            alert('Activity not found. Please search for activities first.');
            return;
        }
    } else {
        activity = activityData;
    }

    // Create planned activity with default time (next available slot)
    const now = new Date();
    const tomorrow = new Date(now);
    tomorrow.setDate(tomorrow.getDate() + 1);
    tomorrow.setHours(10, 0, 0, 0); // Default to 10 AM tomorrow

    const plannedActivity = {
        id: `planned_${Date.now()}_${Math.random()}`,
        activityId: activity.id,
        title: activity.title,
        description: activity.description || activity.how_to_play,
        duration_mins: activity.duration_mins || 60,
        startTime: tomorrow.toISOString(),
        endTime: new Date(tomorrow.getTime() + (activity.duration_mins || 60) * 60000).toISOString(),
        location: activity.indoor_outdoor,
        cost: activity.cost,
        materials_needed: activity.materials_needed,
        activityData: activity // Store full activity data
    };

    plannedActivities.push(plannedActivity);
    savePlannedActivities();

    // Show success message
    showPlannerNotification(`Added "${activity.title}" to your plan!`, 'success');

    // Refresh calendar if visible
    if (document.getElementById('planner-view')?.style.display !== 'none') {
        renderPlannerCalendar();
    }
}

// Remove activity from plan
function removeFromPlan(plannedActivityId) {
    plannedActivities = plannedActivities.filter(a => a.id !== plannedActivityId);
    savePlannedActivities();
    renderPlannerCalendar();
    showPlannerNotification('Activity removed from plan', 'info');
}

// Edit activity time
function editPlannedActivity(plannedActivityId) {
    const activity = plannedActivities.find(a => a.id === plannedActivityId);
    if (!activity) return;

    const currentDate = new Date(activity.startTime);
    const dateStr = currentDate.toISOString().split('T')[0];
    const timeStr = currentDate.toTimeString().slice(0, 5);

    // Show edit modal
    const modal = document.getElementById('editActivityModal');
    if (!modal) {
        createEditModal();
    }

    const editModal = document.getElementById('editActivityModal');
    document.getElementById('editActivityTitle').textContent = activity.title;
    document.getElementById('editActivityDate').value = dateStr;
    document.getElementById('editActivityTime').value = timeStr;
    document.getElementById('editActivityDuration').value = activity.duration_mins;

    editModal.dataset.activityId = plannedActivityId;
    editModal.classList.add('active');
}

// Save edited activity
function savePlannedActivityEdit() {
    const modal = document.getElementById('editActivityModal');
    const activityId = modal.dataset.activityId;
    const activity = plannedActivities.find(a => a.id === activityId);

    if (!activity) return;

    const dateStr = document.getElementById('editActivityDate').value;
    const timeStr = document.getElementById('editActivityTime').value;
    const duration = parseInt(document.getElementById('editActivityDuration').value);

    const startTime = new Date(`${dateStr}T${timeStr}`);
    const endTime = new Date(startTime.getTime() + duration * 60000);

    activity.startTime = startTime.toISOString();
    activity.endTime = endTime.toISOString();
    activity.duration_mins = duration;

    savePlannedActivities();
    renderPlannerCalendar();
    closeEditModal();
    showPlannerNotification('Activity updated!', 'success');
}

// Create edit modal
function createEditModal() {
    const modalHTML = `
        <div class="activity-modal" id="editActivityModal">
            <div class="modal-content" style="max-width: 500px;">
                <button class="modal-close" onclick="closeEditModal()">×</button>
                <div class="modal-body">
                    <h2>Edit Activity Time</h2>
                    <h3 id="editActivityTitle" style="color: var(--color-primary); margin-bottom: 20px;"></h3>

                    <div class="form-group">
                        <label class="form-label">Date</label>
                        <input type="date" id="editActivityDate" class="form-control">
                    </div>

                    <div class="form-group" style="margin-top: 15px;">
                        <label class="form-label">Start Time</label>
                        <input type="time" id="editActivityTime" class="form-control">
                    </div>

                    <div class="form-group" style="margin-top: 15px;">
                        <label class="form-label">Duration (minutes)</label>
                        <input type="number" id="editActivityDuration" class="form-control" min="5" step="5">
                    </div>

                    <div style="display: flex; gap: 10px; margin-top: 25px;">
                        <button class="btn btn--primary btn--full-width" onclick="savePlannedActivityEdit()">
                            Save Changes
                        </button>
                        <button class="btn btn--outline btn--full-width" onclick="closeEditModal()">
                            Cancel
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;

    document.body.insertAdjacentHTML('beforeend', modalHTML);
}

function closeEditModal() {
    const modal = document.getElementById('editActivityModal');
    if (modal) {
        modal.classList.remove('active');
    }
}

// Render planner calendar view
function renderPlannerCalendar() {
    const calendarContainer = document.getElementById('plannerCalendarGrid');
    if (!calendarContainer) return;

    // Get current week
    const today = new Date();
    const startOfWeek = new Date(today);
    startOfWeek.setDate(today.getDate() - today.getDay()); // Start on Sunday
    startOfWeek.setHours(0, 0, 0, 0);

    const days = [];
    for (let i = 0; i < 7; i++) {
        const day = new Date(startOfWeek);
        day.setDate(startOfWeek.getDate() + i);
        days.push(day);
    }

    // Generate calendar HTML
    let calendarHTML = '<div class="calendar-week">';

    days.forEach(day => {
        const dayKey = day.toISOString().split('T')[0];
        const isToday = dayKey === today.toISOString().split('T')[0];

        // Get activities for this day
        const dayActivities = plannedActivities.filter(activity => {
            const activityDate = new Date(activity.startTime).toISOString().split('T')[0];
            return activityDate === dayKey;
        }).sort((a, b) => new Date(a.startTime) - new Date(b.startTime));

        calendarHTML += `
            <div class="calendar-day ${isToday ? 'today' : ''}" data-date="${dayKey}">
                <div class="calendar-day-header">
                    <div class="day-name">${day.toLocaleDateString('en-US', { weekday: 'short' })}</div>
                    <div class="day-number">${day.getDate()}</div>
                </div>
                <div class="calendar-day-body" data-date="${dayKey}">
                    ${dayActivities.length === 0 ? '<div class="empty-day">No activities</div>' : ''}
                    ${dayActivities.map(activity => `
                        <div class="calendar-activity-card" draggable="true" data-activity-id="${activity.id}">
                            <div class="activity-time">${new Date(activity.startTime).toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' })}</div>
                            <div class="activity-name">${activity.title}</div>
                            <div class="activity-duration">${activity.duration_mins} min</div>
                            <div class="activity-actions-mini">
                                <button class="btn-icon" onclick="editPlannedActivity('${activity.id}')" title="Edit time">
                                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                        <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
                                        <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
                                    </svg>
                                </button>
                                <button class="btn-icon" onclick="removeFromPlan('${activity.id}')" title="Remove">
                                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                        <line x1="18" y1="6" x2="6" y2="18"></line>
                                        <line x1="6" y1="6" x2="18" y2="18"></line>
                                    </svg>
                                </button>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    });

    calendarHTML += '</div>';
    calendarContainer.innerHTML = calendarHTML;

    // Setup drag and drop
    setupDragAndDrop();

    // Update count badge
    const countBadge = document.getElementById('plannedActivitiesCount');
    if (countBadge) {
        countBadge.textContent = plannedActivities.length;
        countBadge.style.display = plannedActivities.length > 0 ? 'inline' : 'none';
    }
}

// Setup drag and drop functionality
function setupDragAndDrop() {
    const cards = document.querySelectorAll('.calendar-activity-card');
    const dayBodies = document.querySelectorAll('.calendar-day-body');

    cards.forEach(card => {
        card.addEventListener('dragstart', handleDragStart);
        card.addEventListener('dragend', handleDragEnd);
    });

    dayBodies.forEach(dayBody => {
        dayBody.addEventListener('dragover', handleDragOver);
        dayBody.addEventListener('drop', handleDrop);
        dayBody.addEventListener('dragleave', handleDragLeave);
    });
}

let draggedActivityId = null;

function handleDragStart(e) {
    draggedActivityId = e.target.dataset.activityId;
    e.target.style.opacity = '0.5';
    e.dataTransfer.effectAllowed = 'move';
}

function handleDragEnd(e) {
    e.target.style.opacity = '1';
}

function handleDragOver(e) {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    e.currentTarget.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.currentTarget.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('drag-over');

    const targetDate = e.currentTarget.dataset.date;
    const activity = plannedActivities.find(a => a.id === draggedActivityId);

    if (activity && targetDate) {
        // Update activity to new date (keep same time)
        const oldDate = new Date(activity.startTime);
        const newDate = new Date(targetDate);
        newDate.setHours(oldDate.getHours(), oldDate.getMinutes(), 0, 0);

        const duration = activity.duration_mins || 60;
        activity.startTime = newDate.toISOString();
        activity.endTime = new Date(newDate.getTime() + duration * 60000).toISOString();

        savePlannedActivities();
        renderPlannerCalendar();
        showPlannerNotification('Activity moved!', 'success');
    }
}

// Show notification
function showPlannerNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `planner-notification ${type}`;
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 12px 20px;
        background: ${type === 'success' ? '#2ecc71' : type === 'error' ? '#e74c3c' : '#3498db'};
        color: white;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        z-index: 10000;
        animation: slideInRight 0.3s ease;
    `;

    document.body.appendChild(notification);

    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Export planned activities to ICS (calendar file)
function exportPlannedActivitiesToICS() {
    if (plannedActivities.length === 0) {
        alert('No activities to export. Add activities to your plan first!');
        return;
    }

    let icsContent = `BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//Activity Planner//EN
CALNAME:Activity Plan
`;

    plannedActivities.forEach(activity => {
        const startTime = new Date(activity.startTime);
        const endTime = new Date(activity.endTime);

        // Format dates for ICS (YYYYMMDDTHHMMSS)
        const formatDate = (date) => {
            return date.toISOString().replace(/[-:]/g, '').split('.')[0] + 'Z';
        };

        icsContent += `BEGIN:VEVENT
UID:${activity.id}@activityplanner.com
DTSTAMP:${formatDate(new Date())}
DTSTART:${formatDate(startTime)}
DTEND:${formatDate(endTime)}
SUMMARY:${activity.title}
DESCRIPTION:${(activity.description || '').replace(/\n/g, '\\n')}
LOCATION:${activity.location || ''}
STATUS:CONFIRMED
END:VEVENT
`;
    });

    icsContent += 'END:VCALENDAR';

    // Download file
    const blob = new Blob([icsContent], { type: 'text/calendar' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `activity-plan-${new Date().toISOString().split('T')[0]}.ics`;
    a.click();
    URL.revokeObjectURL(url);

    showPlannerNotification('Calendar exported! Import to Google Calendar, Outlook, etc.', 'success');
}

// Store current search results for easy access
function updateCurrentSearchResults(results) {
    currentSearchResults = results || [];
    console.log('Updated current search results:', currentSearchResults.length);
}

// Initialize planner
function initializePlanner() {
    console.log('Initializing planner with', plannedActivities.length, 'activities');

    // Add CSS for animations
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideInRight {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        @keyframes slideOutRight {
            from { transform: translateX(0); opacity: 1; }
            to { transform: translateX(100%); opacity: 0; }
        }
        .calendar-day.drag-over {
            background: rgba(33, 128, 141, 0.1);
            border: 2px dashed var(--color-primary);
        }
        .calendar-activity-card {
            cursor: grab;
        }
        .calendar-activity-card:active {
            cursor: grabbing;
        }
    `;
    document.head.appendChild(style);
}

// Export functions globally
window.addToPlan = addToPlan;
window.removeFromPlan = removeFromPlan;
window.editPlannedActivity = editPlannedActivity;
window.savePlannedActivityEdit = savePlannedActivityEdit;
window.closeEditModal = closeEditModal;
window.renderPlannerCalendar = renderPlannerCalendar;
window.exportPlannedActivitiesToICS = exportPlannedActivitiesToICS;
window.updateCurrentSearchResults = updateCurrentSearchResults;
window.plannedActivities = plannedActivities;

// Initialize on load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializePlanner);
} else {
    initializePlanner();
}
