        // Sample activities data - will be loaded from CSV
        let sampleActivities = [];

        // CSV parsing function
        function parseCSV(text) {
            const lines = text.split('\n');
            const headers = lines[0].split(',');
            const result = [];

            for (let i = 1; i < lines.length; i++) {
                if (!lines[i].trim()) continue;

                const obj = {};
                const currentLine = parseCSVLine(lines[i]);

                for (let j = 0; j < headers.length; j++) {
                    const header = headers[j].trim();
                    let value = currentLine[j] ? currentLine[j].trim() : '';

                    // Parse special fields
                    if (header === 'age_min' || header === 'age_max' || header === 'duration_mins') {
                        value = parseInt(value) || 0;
                    } else if (header === 'tags' || header === 'materials_needed' || header === 'how_to_play') {
                        // Parse array fields - they come as JSON strings from CSV
                        try {
                            value = JSON.parse(value.replace(/'/g, '"'));
                            if (!Array.isArray(value)) value = [value];
                            // For how_to_play, join array into string
                            if (header === 'how_to_play') {
                                value = Array.isArray(value) ? value.join(' ') : value;
                            }
                        } catch (e) {
                            if (header === 'tags') {
                                value = value ? value.split(',').map(t => t.trim()) : [];
                            } else if (header === 'materials_needed') {
                                value = value ? [value] : [];
                            } else {
                                value = value || '';
                            }
                        }
                    } else if (header === 'season') {
                        // Handle season field
                        value = value.toLowerCase() === 'all' ? ['All seasons'] : [value];
                    } else if (header === 'cost') {
                        // Normalize cost field
                        if (value.toLowerCase() === 'free') value = 'Free';
                        else if (value.toLowerCase() === 'low') value = '$1-10';
                    } else if (header === 'indoor_outdoor') {
                        // Make 'both' more descriptive
                        if (value.toLowerCase() === 'both') value = 'Indoor & Outdoor';
                        else value = value.charAt(0).toUpperCase() + value.slice(1).toLowerCase();
                    } else if (header === 'parent_caution') {
                        value = value === 'no' || value === '' ? '' : 'Parental supervision recommended';
                    }

                    obj[header] = value;
                }

                if (obj.title) result.push(obj);
            }

            return result;
        }

        // Helper function to parse CSV line accounting for quoted fields
        function parseCSVLine(line) {
            const result = [];
            let current = '';
            let inQuotes = false;

            for (let i = 0; i < line.length; i++) {
                const char = line[i];

                if (char === '"') {
                    inQuotes = !inQuotes;
                } else if (char === ',' && !inQuotes) {
                    result.push(current);
                    current = '';
                } else {
                    current += char;
                }
            }
            result.push(current);

            return result;
        }

        // Load activities from API
        async function loadActivitiesFromCSV() {
            try {
                const response = await fetch('/api/activities');
                const data = await response.json();

                if (data.status === 'success' && data.activities) {
                    // Transform API format to match expected structure
                    sampleActivities = data.activities.map(activity => ({
                        ...activity,
                        tags: activity.tags ? (typeof activity.tags === 'string' ? activity.tags.split(',').map(t => t.trim()) : activity.tags) : [],
                        season: activity.season || 'All seasons',
                        materials_needed: activity.materials_needed || 'None'
                    }));
                    console.log(`Loaded ${sampleActivities.length} activities from API`);

                    // Initialize the display
                    currentActivities = [...sampleActivities];
                    renderFeaturedActivities();
                } else {
                    throw new Error('Failed to load activities from API');
                }
            } catch (error) {
                console.error('Error loading activities:', error);
                sampleActivities = [];
                // Don't show alert, just log the error
                console.log('Activities will be loaded on-demand via search');
            }
        }

        // Export to ICS (iCalendar format)
        function exportToICS() {
            if (scheduledActivities.length === 0) {
                alert('No activities scheduled yet. Please add activities to your calendar first.');
                return;
            }

            let icsContent = [
                'BEGIN:VCALENDAR',
                'VERSION:2.0',
                'PRODID:-//Family Activity Planner//EN',
                'CALSCALE:GREGORIAN',
                'METHOD:PUBLISH',
                'X-WR-CALNAME:Family Activities',
                'X-WR-TIMEZONE:UTC'
            ];

            scheduledActivities.forEach(event => {
                const startDate = new Date(event.date);
                const endDate = new Date(startDate.getTime() + (event.duration || 60) * 60000);

                icsContent.push(
                    'BEGIN:VEVENT',
                    `DTSTART:${formatICSDate(startDate)}`,
                    `DTEND:${formatICSDate(endDate)}`,
                    `SUMMARY:${event.title}`,
                    `DESCRIPTION:${event.description || event.how_to_play || ''}`,
                    `LOCATION:${event.indoor_outdoor || ''}`,
                    `UID:${event.id || Date.now()}@familyactivityplanner.com`,
                    `DTSTAMP:${formatICSDate(new Date())}`,
                    'END:VEVENT'
                );
            });

            icsContent.push('END:VCALENDAR');

            const blob = new Blob([icsContent.join('\r\n')], { type: 'text/calendar;charset=utf-8' });
            const link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = 'family-activities.ics';
            link.click();

            console.log('ICS exported successfully');
        }

        // Helper function to format dates for ICS
        function formatICSDate(date) {
            const pad = (n) => n < 10 ? '0' + n : n;
            return date.getUTCFullYear() +
                   pad(date.getUTCMonth() + 1) +
                   pad(date.getUTCDate()) + 'T' +
                   pad(date.getUTCHours()) +
                   pad(date.getUTCMinutes()) +
                   pad(date.getUTCSeconds()) + 'Z';
        }

        // Export to CSV
        function exportToCSV() {
            if (scheduledActivities.length === 0) {
                alert('No activities scheduled yet. Please add activities to your calendar first.');
                return;
            }

            const headers = ['Date', 'Time', 'Activity', 'Duration (mins)', 'Age Range', 'Location', 'Cost', 'Materials', 'Instructions'];
            const rows = scheduledActivities.map(activity => {
                const date = new Date(activity.date);
                return [
                    date.toLocaleDateString(),
                    date.toLocaleTimeString(),
                    activity.title || '',
                    activity.duration_mins || '',
                    `${activity.age_min}-${activity.age_max}` || '',
                    activity.indoor_outdoor || '',
                    activity.cost || '',
                    Array.isArray(activity.materials_needed) ? activity.materials_needed.join('; ') : activity.materials_needed || '',
                    activity.how_to_play || ''
                ].map(field => `"${String(field).replace(/"/g, '""')}"`).join(',');
            });

            const csvContent = [headers.join(','), ...rows].join('\n');
            const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
            const link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = 'family-activities-schedule.csv';
            link.click();

            console.log('CSV exported successfully');
        }

        // Export to PDF
        function exportToPDF() {
            if (scheduledActivities.length === 0) {
                alert('No activities scheduled yet. Please add activities to your calendar first.');
                return;
            }

            // Create a simple text-based PDF using a data URL approach
            const pdf = generatePDFContent();

            // For a real implementation, you'd use jsPDF library
            // For now, we'll create a printable HTML version
            const printWindow = window.open('', '_blank');
            printWindow.document.write(`
                <html>
                <head>
                    <title>Family Activities Schedule</title>
                    <style>
                        body { font-family: Arial, sans-serif; padding: 20px; }
                        h1 { color: #21808D; }
                        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
                        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                        th { background-color: #21808D; color: white; }
                        tr:nth-child(even) { background-color: #f9f9f9; }
                        .activity-details { font-size: 0.9em; color: #666; margin-top: 5px; }
                        @media print {
                            button { display: none; }
                        }
                    </style>
                </head>
                <body>
                    <h1>Family Activities Schedule</h1>
                    <p>Generated on: ${new Date().toLocaleDateString()}</p>
                    <button onclick="window.print()" style="padding: 10px 20px; background: #21808D; color: white; border: none; border-radius: 5px; cursor: pointer; margin-bottom: 20px;">Print / Save as PDF</button>
                    ${pdf}
                </body>
                </html>
            `);
            printWindow.document.close();
        }

        function generatePDFContent() {
            let html = '<table>';
            html += '<thead><tr><th>Date & Time</th><th>Activity</th><th>Details</th></tr></thead>';
            html += '<tbody>';

            scheduledActivities
                .sort((a, b) => new Date(a.date) - new Date(b.date))
                .forEach(activity => {
                    const date = new Date(activity.date);
                    html += `
                        <tr>
                            <td>
                                <strong>${date.toLocaleDateString()}</strong><br>
                                ${date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
                            </td>
                            <td>
                                <strong>${activity.title}</strong>
                                <div class="activity-details">
                                    Ages: ${activity.age_min}-${activity.age_max} |
                                    Duration: ${activity.duration_mins} mins |
                                    ${activity.indoor_outdoor}
                                </div>
                            </td>
                            <td>
                                <div class="activity-details">
                                    <strong>Cost:</strong> ${activity.cost}<br>
                                    ${activity.materials_needed ? `<strong>Materials:</strong> ${Array.isArray(activity.materials_needed) ? activity.materials_needed.join(', ') : activity.materials_needed}<br>` : ''}
                                    ${activity.how_to_play ? `<strong>Instructions:</strong> ${activity.how_to_play}` : ''}
                                </div>
                            </td>
                        </tr>
                    `;
                });

            html += '</tbody></table>';
            return html;
        }

        // Add activity to plan (schedule it)
        function addToPlan(activityTitle) {
            const activity = sampleActivities.find(a => a.title === activityTitle);
            if (!activity) return;

            // Prompt for date and time
            const dateStr = prompt('Enter date (YYYY-MM-DD):', new Date().toISOString().split('T')[0]);
            if (!dateStr) return;

            const timeStr = prompt('Enter time (HH:MM):', '10:00');
            if (!timeStr) return;

            const dateTime = new Date(`${dateStr}T${timeStr}`);

            if (isNaN(dateTime.getTime())) {
                alert('Invalid date or time format');
                return;
            }

            // Add to scheduled activities
            const scheduledActivity = {
                ...activity,
                date: dateTime.toISOString(),
                id: Date.now() + Math.random()
            };

            scheduledActivities.push(scheduledActivity);


            // Render calendar if on calendar view
            renderCalendar();

            console.log('Activity scheduled:', scheduledActivity);
        }

        // Render dynamic calendar
        function renderCalendar() {
            const calendarGrid = document.querySelector('.calendar-grid');
            if (!calendarGrid) return;

            // Get current month
            const now = new Date();
            const year = now.getFullYear();
            const month = now.getMonth();

            // Clear existing calendar (except headers)
            const headers = Array.from(calendarGrid.querySelectorAll('.calendar-header'));
            calendarGrid.innerHTML = '';
            headers.forEach(h => calendarGrid.appendChild(h));

            // Get first day of month and number of days
            const firstDay = new Date(year, month, 1).getDay();
            const daysInMonth = new Date(year, month + 1, 0).getDate();

            // Add empty cells for days before month starts
            for (let i = 0; i < firstDay; i++) {
                const emptyDay = document.createElement('div');
                emptyDay.className = 'calendar-day';
                calendarGrid.appendChild(emptyDay);
            }

            // Add days of month
            for (let day = 1; day <= daysInMonth; day++) {
                const dayElement = document.createElement('div');
                dayElement.className = 'calendar-day';

                const dayNumber = document.createElement('div');
                dayNumber.className = 'day-number';
                dayNumber.textContent = day;
                dayElement.appendChild(dayNumber);

                // Check for scheduled activities on this day
                const dayDate = new Date(year, month, day);
                const dayActivities = scheduledActivities.filter(activity => {
                    const activityDate = new Date(activity.date);
                    return activityDate.getDate() === day &&
                           activityDate.getMonth() === month &&
                           activityDate.getFullYear() === year;
                });

                if (dayActivities.length > 0) {
                    const activitiesContainer = document.createElement('div');
                    activitiesContainer.className = 'day-activities';

                    dayActivities.forEach(activity => {
                        const activityDiv = document.createElement('div');
                        activityDiv.className = 'calendar-activity';
                        activityDiv.draggable = true;
                        activityDiv.dataset.activityId = activity.id;
                        const time = new Date(activity.date).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
                        activityDiv.innerHTML = `
                            <span style="flex: 1;">${activity.title} (${time})</span>
                            <span style="display: inline-flex; gap: 4px;">
                                <button class="btn-icon" onclick="event.stopPropagation(); editScheduledActivity(${activity.id})" title="Edit">✏️</button>
                                <button class="btn-icon" onclick="event.stopPropagation(); deleteScheduledActivity(${activity.id})" title="Delete">🗑️</button>
                            </span>
                        `;
                        activityDiv.style.cursor = 'move';
                        activityDiv.style.display = 'flex';
                        activityDiv.style.alignItems = 'center';
                        activityDiv.style.justifyContent = 'space-between';
                        activityDiv.style.padding = '8px';
                        activityDiv.style.marginBottom = '4px';
                        activityDiv.style.backgroundColor = 'var(--color-primary)';
                        activityDiv.style.color = 'white';
                        activityDiv.style.borderRadius = '4px';

                        // Drag events
                        activityDiv.addEventListener('dragstart', (e) => {
                            e.dataTransfer.effectAllowed = 'move';
                            e.dataTransfer.setData('text/plain', activity.id);
                            activityDiv.style.opacity = '0.5';
                        });

                        activityDiv.addEventListener('dragend', (e) => {
                            activityDiv.style.opacity = '1';
                        });

                        activitiesContainer.appendChild(activityDiv);
                    });

                    dayElement.appendChild(activitiesContainer);
                }

                // Make day element a drop target
                dayElement.addEventListener('dragover', (e) => {
                    e.preventDefault();
                    e.dataTransfer.dropEffect = 'move';
                    dayElement.style.backgroundColor = 'var(--color-secondary)';
                });

                dayElement.addEventListener('dragleave', (e) => {
                    dayElement.style.backgroundColor = '';
                });

                dayElement.addEventListener('drop', (e) => {
                    e.preventDefault();
                    dayElement.style.backgroundColor = '';
                    const activityId = parseFloat(e.dataTransfer.getData('text/plain'));
                    moveActivityToDay(activityId, day, month, year);
                });

                calendarGrid.appendChild(dayElement);
            }
        }

        // Move activity to a different day (drag and drop)
        function moveActivityToDay(activityId, day, month, year) {
            const activity = scheduledActivities.find(a => a.id === activityId);
            if (!activity) return;

            const oldDate = new Date(activity.date);
            const oldTime = oldDate.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});

            // Create new date with same time but different day
            const newDate = new Date(year, month, day, oldDate.getHours(), oldDate.getMinutes());
            activity.date = newDate.toISOString();

            renderCalendar();
            console.log(`Moved "${activity.title}" to ${newDate.toLocaleDateString()}`);
        }

        // Edit scheduled activity
        function editScheduledActivity(activityId) {
            const activity = scheduledActivities.find(a => a.id === activityId);
            if (!activity) return;

            const oldDate = new Date(activity.date);

            // Prompt for new date
            const newDateStr = prompt('Enter new date (YYYY-MM-DD):', oldDate.toISOString().split('T')[0]);
            if (!newDateStr) return;

            // Prompt for new time
            const newTimeStr = prompt('Enter new time (HH:MM):', oldDate.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}));
            if (!newTimeStr) return;

            const newDateTime = new Date(`${newDateStr}T${newTimeStr}`);

            if (isNaN(newDateTime.getTime())) {
                alert('Invalid date or time format');
                return;
            }

            activity.date = newDateTime.toISOString();
            renderCalendar();
            console.log(`Updated "${activity.title}" to ${newDateTime.toLocaleString()}`);
        }

        // Delete scheduled activity
        function deleteScheduledActivity(activityId) {
            const activity = scheduledActivities.find(a => a.id === activityId);
            if (!activity) return;

            if (confirm(`Remove "${activity.title}" from your calendar?`)) {
                scheduledActivities = scheduledActivities.filter(a => a.id !== activityId);
                renderCalendar();
                console.log(`Removed "${activity.title}" from calendar`);
            }
        }

        // View scheduled activity details (kept for compatibility)
        function viewScheduledActivity(activityId) {
            editScheduledActivity(activityId);
        }

        // Application state
        let currentActivities = [...sampleActivities];
        let isLoggedIn = false;
        let isGuestMode = false;
        let currentFilters = {};
        let groupMembers = [];
        window.groupMembers = groupMembers;
        let editingMemberId = null;
        let scheduledActivities = []; // Store scheduled activities with dates and times
        
        // Sample group members data
        const sampleGroupMembers = [
            {
                id: 1,
                name: "Emma Johnson",
                nickname: "Em",
                date_of_birth: "2015-06-15",
                age: 10,
                gender: "Female",
                height_cm: 140,
                weight_kg: 35,
                bmi: 17.9,
                overall_ability: 7,
                motor_skills: { fine: 8, gross: 7 },
                physical_capabilities: { endurance: 6, strength: 5, flexibility: 8, balance: 7 },
                cognitive_abilities: { reading_level: "Advanced", math_skills: "Simple arithmetic", problem_solving: 7 },
                social_skills: { group_interaction: 8, leadership: 6 },
                special_needs: [],
                allergies: "Peanuts",
                medical_conditions: "",
                interests: ["Arts & Crafts", "Reading & Storytelling", "Nature & Outdoors"],
                activity_preference: "Both",
                energy_level: "Moderate"
            },
            {
                id: 2,
                name: "Lucas Johnson",
                nickname: "Luke",
                date_of_birth: "2018-03-22",
                age: 7,
                gender: "Male",
                height_cm: 122,
                weight_kg: 24,
                bmi: 16.1,
                overall_ability: 6,
                motor_skills: { fine: 5, gross: 8 },
                physical_capabilities: { endurance: 7, strength: 6, flexibility: 6, balance: 8 },
                cognitive_abilities: { reading_level: "Beginning", math_skills: "Basic counting", problem_solving: 5 },
                social_skills: { group_interaction: 7, leadership: 5 },
                special_needs: ["ADHD"],
                allergies: "",
                medical_conditions: "",
                interests: ["Sports & Physical Activities", "Building & Construction", "Nature & Outdoors"],
                activity_preference: "Outdoor",
                energy_level: "Very High"
            },
            {
                id: 3,
                name: "Sophie Johnson",
                nickname: "Soph",
                date_of_birth: "2020-11-08",
                age: 4,
                gender: "Female",
                height_cm: 102,
                weight_kg: 16,
                bmi: 15.4,
                overall_ability: 4,
                motor_skills: { fine: 4, gross: 5 },
                physical_capabilities: { endurance: 4, strength: 3, flexibility: 7, balance: 5 },
                cognitive_abilities: { reading_level: "Pre-reader", math_skills: "Basic counting", problem_solving: 3 },
                social_skills: { group_interaction: 6, leadership: 3 },
                special_needs: [],
                allergies: "Dairy",
                medical_conditions: "",
                interests: ["Arts & Crafts", "Music & Performance", "Reading & Storytelling"],
                activity_preference: "Indoor",
                energy_level: "Moderate"
            }
        ];

        // Member Templates Data
        const memberTemplates = {
            template_toddler: {
                id: "template_toddler",
                name: "Toddler",
                age_range: "2-3 years",
                age: 3,
                date_of_birth: "2022-01-15",
                gender: "Prefer not to say",
                height_cm: 95,
                weight_kg: 14,
                overall_ability: 2,
                motor_skills: { fine: 2, gross: 3 },
                physical_capabilities: { endurance: 2, strength: 2, flexibility: 5, balance: 3 },
                cognitive_abilities: { reading_level: "Pre-reader", math_skills: "Basic counting", problem_solving: 2 },
                social_skills: { group_interaction: 4, leadership: 2 },
                special_needs: [],
                allergies: "",
                interests: ["Arts & Crafts", "Music & Performance"],
                activity_preference: "Indoor",
                energy_level: "Moderate",
                icon: "👶"
            },
            template_preschooler: {
                id: "template_preschooler",
                name: "Preschooler",
                age_range: "4-5 years",
                age: 5,
                date_of_birth: "2020-06-01",
                gender: "Prefer not to say",
                height_cm: 108,
                weight_kg: 18,
                overall_ability: 4,
                motor_skills: { fine: 4, gross: 5 },
                physical_capabilities: { endurance: 4, strength: 3, flexibility: 6, balance: 5 },
                cognitive_abilities: { reading_level: "Pre-reader", math_skills: "Basic counting", problem_solving: 3 },
                social_skills: { group_interaction: 5, leadership: 3 },
                special_needs: [],
                allergies: "",
                interests: ["Arts & Crafts", "Nature & Outdoors", "Music & Performance"],
                activity_preference: "Both",
                energy_level: "High",
                icon: "🧒"
            },
            template_early_elem: {
                id: "template_early_elem",
                name: "Early Elementary",
                age_range: "6-8 years",
                age: 7,
                date_of_birth: "2018-03-15",
                gender: "Prefer not to say",
                height_cm: 122,
                weight_kg: 24,
                overall_ability: 6,
                motor_skills: { fine: 6, gross: 7 },
                physical_capabilities: { endurance: 6, strength: 5, flexibility: 7, balance: 7 },
                cognitive_abilities: { reading_level: "Beginning", math_skills: "Simple arithmetic", problem_solving: 5 },
                social_skills: { group_interaction: 7, leadership: 5 },
                special_needs: [],
                allergies: "",
                interests: ["Sports & Physical Activities", "Science & Exploration", "Building & Construction"],
                activity_preference: "Outdoor",
                energy_level: "Very High",
                icon: "👧"
            },
            template_late_elem: {
                id: "template_late_elem",
                name: "Late Elementary",
                age_range: "9-11 years",
                age: 10,
                date_of_birth: "2015-09-01",
                gender: "Prefer not to say",
                height_cm: 140,
                weight_kg: 35,
                overall_ability: 7,
                motor_skills: { fine: 8, gross: 8 },
                physical_capabilities: { endurance: 7, strength: 6, flexibility: 7, balance: 8 },
                cognitive_abilities: { reading_level: "Advanced", math_skills: "Advanced math", problem_solving: 7 },
                social_skills: { group_interaction: 8, leadership: 7 },
                special_needs: [],
                allergies: "",
                interests: ["Science & Exploration", "Technology & Gaming", "Sports & Physical Activities"],
                activity_preference: "Both",
                energy_level: "High",
                icon: "🧑"
            },
            template_preteen: {
                id: "template_preteen",
                name: "Preteen",
                age_range: "12-14 years",
                age: 13,
                date_of_birth: "2012-05-20",
                gender: "Prefer not to say",
                height_cm: 158,
                weight_kg: 48,
                overall_ability: 8,
                motor_skills: { fine: 9, gross: 8 },
                physical_capabilities: { endurance: 8, strength: 7, flexibility: 7, balance: 8 },
                cognitive_abilities: { reading_level: "Advanced", math_skills: "Advanced math", problem_solving: 8 },
                social_skills: { group_interaction: 8, leadership: 8 },
                special_needs: [],
                allergies: "",
                interests: ["Technology & Gaming", "Sports & Physical Activities", "Music & Performance"],
                activity_preference: "Both",
                energy_level: "High",
                icon: "👦"
            },
            template_teen: {
                id: "template_teen",
                name: "Teen",
                age_range: "15-17 years",
                age: 16,
                date_of_birth: "2009-02-10",
                gender: "Prefer not to say",
                height_cm: 170,
                weight_kg: 62,
                overall_ability: 9,
                motor_skills: { fine: 9, gross: 9 },
                physical_capabilities: { endurance: 9, strength: 8, flexibility: 7, balance: 9 },
                cognitive_abilities: { reading_level: "Advanced", math_skills: "Advanced math", problem_solving: 9 },
                social_skills: { group_interaction: 8, leadership: 9 },
                special_needs: [],
                allergies: "",
                interests: ["Sports & Physical Activities", "Technology & Gaming", "Science & Exploration"],
                activity_preference: "Both",
                energy_level: "Very High",
                icon: "🧑‍🎓"
            },
            template_adult: {
                id: "template_adult",
                name: "Adult Supervisor",
                age_range: "25-45 years",
                age: 35,
                date_of_birth: "1990-07-15",
                gender: "Prefer not to say",
                height_cm: 170,
                weight_kg: 70,
                overall_ability: 8,
                motor_skills: { fine: 9, gross: 8 },
                physical_capabilities: { endurance: 7, strength: 7, flexibility: 6, balance: 8 },
                cognitive_abilities: { reading_level: "Advanced", math_skills: "Advanced math", problem_solving: 9 },
                social_skills: { group_interaction: 9, leadership: 10 },
                special_needs: [],
                allergies: "",
                interests: ["Nature & Outdoors", "Sports & Physical Activities", "Arts & Crafts"],
                activity_preference: "Both",
                energy_level: "Moderate",
                icon: "👨‍👩"
            },
            template_senior: {
                id: "template_senior",
                name: "Senior",
                age_range: "65+ years",
                age: 70,
                date_of_birth: "1955-04-20",
                gender: "Prefer not to say",
                height_cm: 165,
                weight_kg: 68,
                overall_ability: 5,
                motor_skills: { fine: 7, gross: 5 },
                physical_capabilities: { endurance: 4, strength: 4, flexibility: 4, balance: 5 },
                cognitive_abilities: { reading_level: "Advanced", math_skills: "Advanced math", problem_solving: 9 },
                social_skills: { group_interaction: 9, leadership: 8 },
                special_needs: [],
                allergies: "",
                interests: ["Reading & Storytelling", "Nature & Outdoors", "Arts & Crafts"],
                activity_preference: "Both",
                energy_level: "Low",
                icon: "👴"
            }
        };
        
        // Template functions
        function addMemberTemplate(templateId) {
            const template = memberTemplates[templateId];
            if (!template) {
                alert('Template not found!');
                return;
            }
            
            // Create a new member from template with property mapping
            const memberCount = groupMembers.filter(m => m.name.includes(template.name)).length;
            const newMember = {
                ...template,
                id: Date.now(), // New unique ID
                name: memberCount > 0 ? `${template.name} ${memberCount + 1}` : template.name,
                bmi: calculateBMI(template.height_cm, template.weight_kg),
                medical_conditions: "",
                // Map overall_ability to ability for API compatibility
                ability: template.overall_ability,
                physical_ability: template.overall_ability
            };
            
            groupMembers.push(newMember);
            
            window.groupMembers = groupMembers;
            
            console.log('Added template member:', newMember);
            console.log('Total group members:', groupMembers.length);
            console.log('Total window.groupMembers:', window.groupMembers.length);

            renderGroupMembers();
        }

        // DOM elements
        const views = document.querySelectorAll('.view');
        const navLinks = document.querySelectorAll('.nav-link');
        const loginNavBtn = document.getElementById('loginNavBtn');
        const logoutBtn = document.getElementById('logoutBtn');
        const userInfo = document.getElementById('userInfo');

        // Helper functions
        function calculateAge(birthDate) {
            const today = new Date();
            const birth = new Date(birthDate);
            let age = today.getFullYear() - birth.getFullYear();
            const monthDiff = today.getMonth() - birth.getMonth();
            if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birth.getDate())) {
                age--;
            }
            return age;
        }
        
        function calculateBMI(heightCm, weightKg) {
            const heightM = heightCm / 100;
            return Math.round((weightKg / (heightM * heightM)) * 10) / 10;
        }
        
        function getAbilityLevelText(level) {
            if (level <= 2) return "Beginner";
            if (level <= 4) return "Developing";
            if (level <= 6) return "Intermediate";
            if (level <= 8) return "Advanced";
            return "Expert";
        }
        
        function getInitials(name) {
            return name.split(' ').map(word => word[0]).join('').toUpperCase();
        }
        
        // Group member management
        function renderGroupMembers() {
            const grid = document.getElementById('membersGrid');
            const emptyState = document.getElementById('emptyState');
            
            if (groupMembers.length === 0) {
                emptyState.style.display = 'block';
                grid.style.display = 'none';
            } else {
                emptyState.style.display = 'none';
                grid.style.display = 'grid';
                grid.innerHTML = groupMembers.map(member => createMemberCard(member)).join('');
            }
            
            updateGroupSummary();
        }
        
        function createMemberCard(member) {
            const abilityText = getAbilityLevelText(member.overall_ability);
            const abilityColorValues = member.overall_ability <= 4 ? '168, 75, 47' : member.overall_ability <= 7 ? '98, 108, 113' : '33, 128, 141';
            
            return `
                <div class="card member-card" style="padding: var(--space-16);">
                    <div style="display: flex; align-items: flex-start; gap: var(--space-12); margin-bottom: var(--space-16);">
                        <div style="width: 50px; height: 50px; border-radius: var(--radius-full); background: var(--color-primary); color: var(--color-btn-primary-text); display: flex; align-items: center; justify-content: center; font-weight: var(--font-weight-bold); font-size: var(--font-size-lg); flex-shrink: 0;">
                            ${getInitials(member.name)}
                        </div>
                        <div style="flex: 1; min-width: 0;">
                            <h4 style="margin: 0 0 var(--space-4) 0; word-wrap: break-word;">${member.name}</h4>
                            ${member.nickname ? `<p class="secondary-info" style="margin: 0 0 var(--space-4) 0; color: var(--color-text-secondary); font-size: var(--font-size-sm);">"${member.nickname}"</p>` : ''}
                            <p style="margin: 0; color: var(--color-text-secondary); font-size: var(--font-size-sm);">${member.age} years old</p>
                        </div>
                    </div>
                    
                    <div class="secondary-info" style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-8); margin-bottom: var(--space-16); font-size: var(--font-size-sm);">
                        <div>
                            <strong>Height:</strong> ${member.height_cm}cm
                        </div>
                        <div>
                            <strong>Weight:</strong> ${member.weight_kg}kg
                        </div>
                        <div style="grid-column: 1 / -1;">
                            <strong>BMI:</strong> ${member.bmi}
                        </div>
                    </div>
                    
                    <div style="margin-bottom: var(--space-16);">
                        <div class="status" style="background: rgba(${abilityColorValues}, 0.15); color: rgb(${abilityColorValues}); border-color: rgba(${abilityColorValues}, 0.25);">
                            ${abilityText} (${member.overall_ability}/10)
                        </div>
                    </div>
                    
                    ${member.special_needs.length > 0 ? `
                        <div style="margin-bottom: var(--space-16); font-size: var(--font-size-sm);">
                            <strong>Special Needs:</strong> ${member.special_needs.join(', ')}
                        </div>
                    ` : ''}
                    
                    ${member.allergies ? `
                        <div style="margin-bottom: var(--space-16); font-size: var(--font-size-sm); color: var(--color-warning);">
                            <strong>⚠️ Allergies:</strong> ${member.allergies}
                        </div>
                    ` : ''}
                    
                    <div class="secondary-info" style="margin-bottom: var(--space-16); font-size: var(--font-size-sm);">
                        <strong>Interests:</strong> ${member.interests.slice(0, 3).join(', ')}${member.interests.length > 3 ? '...' : ''}
                    </div>
                    
                    <div class="flex gap-8" style="flex-wrap: wrap;">
                        <button class="btn btn--outline btn--sm" onclick="editMemberPage(${member.id})">Edit</button>
                        <button class="btn btn--outline btn--sm" onclick="duplicateMember(${member.id})">Duplicate</button>
                        <button class="btn btn--outline btn--sm" style="color: var(--color-error); border-color: var(--color-error);" onclick="removeMember(${member.id})">Remove</button>
                    </div>
                </div>
            `;
        }
        
        function updateGroupSummary() {
            const totalMembers = document.getElementById('totalMembers');
            const ageRange = document.getElementById('ageRange');
            const avgAbility = document.getElementById('avgAbility');
            const getRecommendationsBtn = document.getElementById('getRecommendationsBtn');
            
            totalMembers.textContent = groupMembers.length;
            
            if (groupMembers.length === 0) {
                ageRange.textContent = '-';
                avgAbility.textContent = '-';
                getRecommendationsBtn.disabled = true;
            } else {
                const ages = groupMembers.map(m => m.age);
                const minAge = Math.min(...ages);
                const maxAge = Math.max(...ages);
                ageRange.textContent = minAge === maxAge ? `${minAge}` : `${minAge}-${maxAge}`;
                
                const totalAbility = groupMembers.reduce((sum, m) => sum + m.overall_ability, 0);
                avgAbility.textContent = Math.round(totalAbility / groupMembers.length * 10) / 10;
                
                getRecommendationsBtn.disabled = false;
            }
        }
        
        function editMemberPage(memberId) {
            editingMemberId = memberId;
            const member = groupMembers.find(m => m.id === memberId);
            
            // Update page title and breadcrumb
            document.getElementById('memberFormTitle').textContent = 'Edit Member';
            document.getElementById('memberFormBreadcrumb').textContent = 'Edit Member';
            
            // Populate form with member data
            const form = document.getElementById('memberPageForm');
            populateMemberPageForm(form, member);
            
            showView('member-form');
        }
        
        function duplicateMember(memberId) {
            const memberToDuplicate = groupMembers.find(m => m.id === memberId);
            if (!memberToDuplicate) {
                alert('Member not found!');
                return;
            }
            
            // Create a copy with new ID and modified name
            const duplicatedMember = {
                ...memberToDuplicate,
                id: Date.now(),
                name: memberToDuplicate.name + ' (Copy)',
                nickname: memberToDuplicate.nickname ? memberToDuplicate.nickname + ' Copy' : ''
            };
            
            groupMembers.push(duplicatedMember);
            renderGroupMembers();
            
        }
        
        function populateMemberPageForm(form, member) {
            const formData = new FormData(form);
            
            // Basic info
            form.name.value = member.name;
            form.nickname.value = member.nickname || '';
            form.date_of_birth.value = member.date_of_birth;
            form.gender.value = member.gender || '';
            
            // Physical characteristics
            form.height_cm.value = member.height_cm;
            form.weight_kg.value = member.weight_kg;
            form.bmi.value = member.bmi;
            
            // Abilities
            form.overall_ability.value = member.overall_ability;
            form.fine_motor.value = member.motor_skills?.fine || 5;
            form.gross_motor.value = member.motor_skills?.gross || 5;
            form.endurance.value = member.physical_capabilities?.endurance || 5;
            form.strength.value = member.physical_capabilities?.strength || 5;
            form.flexibility.value = member.physical_capabilities?.flexibility || 5;
            form.balance.value = member.physical_capabilities?.balance || 5;
            form.reading_level.value = member.cognitive_abilities?.reading_level || 'Pre-reader';
            form.math_skills.value = member.cognitive_abilities?.math_skills || 'Basic counting';
            form.problem_solving.value = member.cognitive_abilities?.problem_solving || 5;
            form.group_interaction.value = member.social_skills?.group_interaction || 5;
            form.leadership.value = member.social_skills?.leadership || 5;
            
            // Special considerations
            const specialNeedsCheckboxes = form.querySelectorAll('input[name="special_needs"]');
            specialNeedsCheckboxes.forEach(checkbox => {
                checkbox.checked = member.special_needs?.includes(checkbox.value) || false;
            });
            
            form.allergies.value = member.allergies || '';
            form.medical_conditions.value = member.medical_conditions || '';
            
            // Interests
            const interestCheckboxes = form.querySelectorAll('input[name="interests"]');
            interestCheckboxes.forEach(checkbox => {
                checkbox.checked = member.interests?.includes(checkbox.value) || false;
            });
            
            form.activity_preference.value = member.activity_preference || 'Both';
            form.energy_level.value = member.energy_level || 'Moderate';
        }
        
        function saveMemberFromPage(formData) {
            const member = {
                id: editingMemberId || Date.now(),
                name: formData.get('name'),
                nickname: formData.get('nickname'),
                date_of_birth: formData.get('date_of_birth'),
                age: calculateAge(formData.get('date_of_birth')),
                gender: formData.get('gender'),
                height_cm: parseInt(formData.get('height_cm')),
                weight_kg: parseFloat(formData.get('weight_kg')),
                bmi: calculateBMI(parseInt(formData.get('height_cm')), parseFloat(formData.get('weight_kg'))),
                overall_ability: parseInt(formData.get('overall_ability')),
                ability: parseInt(formData.get('overall_ability')),
                physical_ability: parseInt(formData.get('overall_ability')),
                motor_skills: {
                    fine: parseInt(formData.get('fine_motor')),
                    gross: parseInt(formData.get('gross_motor'))
                },
                physical_capabilities: {
                    endurance: parseInt(formData.get('endurance')),
                    strength: parseInt(formData.get('strength')),
                    flexibility: parseInt(formData.get('flexibility')),
                    balance: parseInt(formData.get('balance'))
                },
                cognitive_abilities: {
                    reading_level: formData.get('reading_level'),
                    math_skills: formData.get('math_skills'),
                    problem_solving: parseInt(formData.get('problem_solving'))
                },
                social_skills: {
                    group_interaction: parseInt(formData.get('group_interaction')),
                    leadership: parseInt(formData.get('leadership'))
                },
                special_needs: formData.getAll('special_needs').concat(
                    formData.get('other_special_needs') ? [formData.get('other_special_needs')] : []
                ),
                allergies: formData.get('allergies'),
                medical_conditions: formData.get('medical_conditions'),
                interests: formData.getAll('interests'),
                activity_preference: formData.get('activity_preference'),
                energy_level: formData.get('energy_level')
            };
            
            if (editingMemberId) {
                const index = groupMembers.findIndex(m => m.id === editingMemberId);
                groupMembers[index] = member;
                window.groupMembers = groupMembers;
            } else {
                groupMembers.push(member);
                window.groupMembers = groupMembers;
            }
            
            // ADD THESE LINES to refresh the display and show success
            renderGroupMembers();
            
            // Reset the form
            document.getElementById('memberPageForm').reset();
            
            // Reset editing state
            editingMemberId = null;
            
            // Show success message
            alert('Member saved successfully!');
            
            // Switch back to My Group view to see the new member
            showView('my-group');
        }
        
        function setupMemberPageForm() {
            // Reset form for new member
            editingMemberId = null;
            document.getElementById('memberFormTitle').textContent = 'Add New Member';
            document.getElementById('memberFormBreadcrumb').textContent = 'Add Member';
            document.getElementById('memberPageForm').reset();
        }
        
        function removeMember(memberId) {
            if (confirm('Are you sure you want to remove this member?')) {
                groupMembers = groupMembers.filter(m => m.id !== memberId);
                window.groupMembers = groupMembers;
                renderGroupMembers();
            }
        }
        
        // Note: getPersonalizedRecommendations() is now defined in api-integration.js
        // This function has been removed to prevent override of the real API integration
        
        // Navigation
        function showView(viewId) {
            views.forEach(view => view.classList.remove('active'));
            navLinks.forEach(link => link.classList.remove('active'));

            document.getElementById(viewId).classList.add('active');
            const activeLink = document.querySelector(`[data-view="${viewId}"]`);
            if (activeLink) activeLink.classList.add('active');

            // Load view-specific content
            if (viewId === 'home') {
                renderFeaturedActivities();
            } else if (viewId === 'search') {
                renderActivities(currentActivities);
                updateMyGroupIndicator();
            } else if (viewId === 'calendar') {
                renderCalendar();
            } else if (viewId === 'my-group') {
                renderGroupMembers();
            } else if (viewId === 'member-form') {
                if (!editingMemberId) {
                    setupMemberPageForm();
                }
                // Setup event listeners after a small delay to ensure DOM is ready
                setTimeout(setupMemberFormEventListeners, 100);
            }
        }

        // Event listeners for navigation
        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const viewId = link.dataset.view;
                
                // Check if user needs to be logged in (Guest Mode users can access all views)
                if ((viewId === 'add-activity' || viewId === 'profile' || viewId === 'my-group') && !isLoggedIn && !isGuestMode) {
                    showView('login');
                    return;
                }
                
                showView(viewId);
            });
        });

        // Helper function to normalize indoor_outdoor display
        function normalizeIndoorOutdoor(value) {
            if (!value) return 'Indoor & Outdoor';
            const lower = value.toLowerCase();
            if (lower === 'both') return 'Indoor & Outdoor';
            return value.charAt(0).toUpperCase() + value.slice(1).toLowerCase();
        }

        // Activity card rendering
        function createActivityCard(activity, includeActions = true) {
            const seasonText = Array.isArray(activity.season) ? activity.season.join(', ') : activity.season;
            const materialsText = Array.isArray(activity.materials_needed) ? activity.materials_needed.join(', ') : activity.materials_needed;
            const indoorOutdoorText = normalizeIndoorOutdoor(activity.indoor_outdoor);

            return `
                <div class="activity-card">
                    <div class="activity-header">
                        <h3 class="activity-title">${activity.title}</h3>
                        <div class="meta-badge">${indoorOutdoorText}</div>
                    </div>
                    <div class="activity-meta">
                        <div class="meta-badge">Ages ${activity.age_min}-${activity.age_max}</div>
                        <div class="meta-badge">${activity.duration_mins} mins</div>
                        <div class="meta-badge">${activity.cost}</div>
                        <div class="meta-badge">${activity.players} players</div>
                    </div>
                    <div class="activity-tags">
                        ${activity.tags.map(tag => `<span class="tag">${tag}</span>`).join('')}
                    </div>
                    ${includeActions ? `
                    <div class="activity-actions">
                        <button class="btn btn--outline btn--sm" onclick="viewActivityDetails('${activity.title}')">View Details</button>
                        <button class="btn btn--primary btn--sm" onclick="addToPlan('${activity.title}')">Add to Plan</button>
                    </div>
                    ` : ''}
                </div>
            `;
        }

        function renderActivities(activities) {
            const grid = document.getElementById('activitiesGrid');
            if (activities.length === 0) {
                grid.innerHTML = '<p>No activities found matching your criteria.</p>';
                return;
            }
            grid.innerHTML = activities.map(activity => createActivityCard(activity)).join('');
        }

        function renderFeaturedActivities() {
            const grid = document.getElementById('featuredActivities');
            grid.innerHTML = sampleActivities.slice(0, 3).map(activity => createActivityCard(activity)).join('');
        }

        // Activity details modal
        // View activity details - uses modal
        async function viewActivityDetails(titleOrId, title) {
            // If called with title only (old way), find the activity first
            let activityId = titleOrId;
            if (typeof titleOrId === 'string') {
                const activity = sampleActivities.find(a => a.title === titleOrId);
                if (!activity) {
                    alert('Activity not found');
                    return;
                }
                activityId = activity.id;
            }

            const modal = document.getElementById('activityModal');
            const modalBody = document.getElementById('modalBody');

            if (!modal || !modalBody) {
                console.error('Modal elements not found');
                return;
            }

            // Show modal with loading state
            modal.classList.add('active');
            modalBody.innerHTML = \`
                <div class="loading">
                    <div class="spinner"></div>
                    <p>Loading activity details...</p>
                </div>
            \`;

            try {
                const response = await fetch(\`/api/activity/\${activityId}\`);
                const data = await response.json();

                if (data.status === 'success' && data.activity) {
                    displayActivityDetailsInModal(data.activity);
                } else {
                    modalBody.innerHTML = \`
                        <div class="empty-state">
                            <div class="empty-state-icon">❌</div>
                            <h3>Error Loading Details</h3>
                            <p>\${data.error || data.message || 'Unable to load activity details'}</p>
                        </div>
                    \`;
                }
            } catch (error) {
                console.error('Error fetching activity details:', error);
                modalBody.innerHTML = \`
                    <div class="empty-state">
                        <div class="empty-state-icon">⚠️</div>
                        <h3>Connection Error</h3>
                        <p>Failed to load activity details. Please check your connection and try again.</p>
                    </div>
                \`;
            }
        }

        // Display activity details in modal
        function displayActivityDetailsInModal(activity) {
            const modalBody = document.getElementById('modalBody');

            // Parse materials if it's a string
            let materials = activity.materials_needed;
            if (typeof materials === 'string') {
                materials = materials.split(',').map(m => m.trim());
            }

            // Parse tags if it's a string
            let tags = activity.tags;
            if (typeof tags === 'string') {
                tags = tags.split(',').map(t => t.trim());
            }

            // Normalize indoor_outdoor display
            const indoorOutdoorText = normalizeIndoorOutdoor(activity.indoor_outdoor);

            modalBody.innerHTML = \`
                <div class="modal-header">
                    <h2 class="modal-title">\${activity.title}</h2>
                    <div class="modal-meta">
                        <span class="meta-tag">👥 \${activity.players || 'Any number of players'}</span>
                        <span class="meta-tag">⏱️ \${activity.duration_mins} minutes</span>
                        <span class="meta-tag">🎂 Ages \${activity.age_min}-\${activity.age_max}</span>
                        <span class="meta-tag">💰 \${activity.cost || 'Free'}</span>
                        <span class="meta-tag">📍 \${indoorOutdoorText}</span>
                        \${activity.season ? \`<span class="meta-tag">🌤️ \${activity.season}</span>\` : ''}
                    </div>
                </div>

                <div class="modal-section">
                    <h3>Description</h3>
                    <p>\${activity.description || 'No description available.'}</p>
                </div>

                \${activity.how_to_play ? \`
                    <div class="modal-section">
                        <h3>How to Play / Instructions</h3>
                        <p>\${activity.how_to_play}</p>
                    </div>
                \` : ''}

                \${materials && materials.length > 0 ? \`
                    <div class="modal-section">
                        <h3>Materials Needed</h3>
                        <ul>
                            \${Array.isArray(materials) ? materials.map(m => \`<li>\${m}</li>\`).join('') : \`<li>\${materials}</li>\`}
                        </ul>
                    </div>
                \` : ''}

                \${tags && tags.length > 0 ? \`
                    <div class="modal-section">
                        <h3>Tags & Categories</h3>
                        <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                            \${Array.isArray(tags) ? tags.map(tag => \`<span class="meta-tag">\${tag}</span>\`).join('') : \`<span class="meta-tag">\${tags}</span>\`}
                        </div>
                    </div>
                \` : ''}

                \${activity.parent_caution ? \`
                    <div class="modal-section">
                        <h3>⚠️ Parent Caution</h3>
                        <p style="color: var(--color-error);"><strong>\${activity.parent_caution}</strong></p>
                    </div>
                \` : ''}

                <div class="modal-section">
                    <h3>Quick Details</h3>
                    <div class="detail-grid">
                        <div class="detail-item">
                            <div class="detail-label">Duration</div>
                            <div class="detail-value">\${activity.duration_mins} min</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Age Range</div>
                            <div class="detail-value">\${activity.age_min}-\${activity.age_max} yrs</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Players</div>
                            <div class="detail-value">\${activity.players || 'Any'}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Cost</div>
                            <div class="detail-value">\${activity.cost || 'Free'}</div>
                        </div>
                    </div>
                </div>
            \`;
        }


        // addToPlan function is defined earlier in the file (line ~287)

        function editActivity(title) {
            alert(`Edit functionality for "${title}" would be implemented here.`);
        }

        // Login/logout functionality
        function login() {
            isLoggedIn = true;
            isGuestMode = false;
            loginNavBtn.classList.add('hidden');
            userInfo.classList.remove('hidden');
            document.getElementById('guestInfo').classList.add('hidden');
            document.getElementById('myGroupNavLink').style.display = 'block';
            const guestBanner = document.getElementById('guestModeBanner');
            if (guestBanner) {
                guestBanner.classList.add('hidden');
            }

            // Load sample data for demo
            if (groupMembers.length === 0) {
                groupMembers = [...sampleGroupMembers];
            }

            showView('home');
        }

        function logout() {
            isLoggedIn = false;
            userInfo.classList.add('hidden');
            loginNavBtn.classList.remove('hidden');
            document.getElementById('myGroupNavLink').style.display = 'block';
            
            // Auto-switch to guest mode
            startGuestMode();
        }
        
        function startGuestMode() {
            isGuestMode = true;
            isLoggedIn = false;
            const dashboardElement = document.getElementById('dashboard');
            if (dashboardElement) {
                dashboardElement.classList.add('active');
            }
            console.log('Guest mode activated');
            // Update UI for guest mode
            loginNavBtn.classList.add('hidden');
            const guestInfo = document.getElementById('guestInfo');
            if (guestInfo) {
                guestInfo.classList.remove('hidden');
            }
            const userInfo = document.getElementById('userInfo');
            if (userInfo) {
                userInfo.classList.add('hidden');
            }
            document.getElementById('myGroupNavLink').style.display = 'block';
            const guestBanner = document.getElementById('guestModeBanner');
            if (guestBanner) {
                guestBanner.classList.remove('hidden');
            }

            // Pre-populate with sample data if empty
            if (groupMembers.length === 0) {
                groupMembers = [...sampleGroupMembers];
            }

            console.log('Guest mode started with', groupMembers.length, 'sample members');
        }

        // Auth tab switching
        function switchAuthTab(tabName) {
            const tabs = document.querySelectorAll('.auth-tab');
            const contents = document.querySelectorAll('.auth-content');
            
            tabs.forEach(tab => {
                tab.classList.remove('active');
                tab.style.borderBottomColor = 'transparent';
                tab.style.color = 'var(--color-text-secondary)';
            });
            
            contents.forEach(content => {
                content.classList.add('hidden');
            });
            
            const activeTab = document.querySelector(`[data-tab="${tabName}"]`);
            const activeContent = document.getElementById(`${tabName}TabContent`);
            
            activeTab.classList.add('active');
            activeTab.style.borderBottomColor = 'var(--color-primary)';
            activeTab.style.color = 'var(--color-primary)';
            activeContent.classList.remove('hidden');
            
            // Update switch text
            const switchText = document.getElementById('switchText');
            const switchLink = document.getElementById('switchAuthMode');
            
            if (tabName === 'login') {
                switchText.innerHTML = 'Don\'t have an account? <a href="#" id="switchAuthMode" style="color: var(--color-primary); font-weight: var(--font-weight-medium);">Register</a>';
            } else {
                switchText.innerHTML = 'Already have an account? <a href="#" id="switchAuthMode" style="color: var(--color-primary); font-weight: var(--font-weight-medium);">Login</a>';
            }
            
            // Re-attach event listener to new link
            document.getElementById('switchAuthMode').addEventListener('click', (e) => {
                e.preventDefault();
                switchAuthTab(tabName === 'login' ? 'register' : 'login');
            });
        }

        // Search functionality - Google-like instant search using AI
        async function performSearch(query = '') {
            console.log('🔍 Performing search for:', query);

            // If no query and we have sample activities, show them filtered by current filters
            if (!query && sampleActivities.length > 0) {
                const minAge = parseInt(document.getElementById('minAge')?.value || 0);
                const maxAge = parseInt(document.getElementById('maxAge')?.value || 100);
                const duration = document.getElementById('durationFilter')?.value;
                const cost = document.getElementById('costFilter')?.value;
                const season = document.getElementById('seasonFilter')?.value;
                const indoorOutdoor = document.querySelector('.toggle-btn.active[data-filter="indoor_outdoor"]')?.dataset.value;

                let filtered = sampleActivities.filter(activity => {
                    if (activity.age_min > maxAge || activity.age_max < minAge) return false;
                    if (duration && activity.duration_mins < parseInt(duration)) return false;
                    if (cost && activity.cost !== cost) return false;
                    // Handle Indoor & Outdoor filter - "Indoor & Outdoor" activities match both Indoor and Outdoor filters
                    if (indoorOutdoor) {
                        const activityLocation = activity.indoor_outdoor || '';
                        if (activityLocation !== indoorOutdoor && activityLocation !== 'Indoor & Outdoor' && activityLocation.toLowerCase() !== 'both') {
                            return false;
                        }
                    }
                    if (season) {
                        const activitySeasons = Array.isArray(activity.season) ? activity.season : [activity.season];
                        if (!activitySeasons.includes(season) && !activitySeasons.includes('All seasons')) return false;
                    }
                    return true;
                });

                currentActivities = filtered;
                renderActivities(filtered);
                return;
            }

            // If there's a query, use AI search API
            if (query) {
                try {
                    const response = await fetch('/api/recommend', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            query: query,
                            group_members: window.groupMembers || [],
                            preferences: [],
                            top_k: 20
                        })
                    });

                    const data = await response.json();
                    console.log('📊 API Response:', data);

                    if (data.status === 'success' && data.recommendations) {
                        // Transform API recommendations to match expected structure
                        const activities = data.recommendations.map(rec => ({
                            id: rec.id,
                            title: rec.title || 'Activity',
                            description: rec.description || '',
                            tags: rec.tags ? (typeof rec.tags === 'string' ? rec.tags.split(',').map(t => t.trim()) : rec.tags) : [],
                            age_min: rec.age_min || 0,
                            age_max: rec.age_max || 100,
                            duration_mins: rec.duration_mins || 30,
                            cost: rec.cost || 'Free',
                            indoor_outdoor: rec.indoor_outdoor || rec.location || 'Indoor & Outdoor',
                            players: rec.players || '1+',
                            how_to_play: rec.how_to_play || rec.description || 'No instructions available',
                            materials_needed: rec.materials_needed || 'None',
                            season: rec.season || 'All seasons',
                            parent_caution: rec.parent_caution || '',
                            recommendation_score: rec.recommendation_score || 0
                        }));

                        console.log(`✅ Displaying ${activities.length} activities`);
                        currentActivities = activities;
                        sampleActivities = activities; // Update sample activities for viewing details
                        renderActivities(activities);
                    } else {
                        console.error('❌ Search failed:', data);
                        const grid = document.getElementById('activitiesGrid');
                        if (grid) {
                            grid.innerHTML = '<p>No activities found. Try a different search term.</p>';
                        }
                    }
                } catch (error) {
                    console.error('❌ Search error:', error);
                    const grid = document.getElementById('activitiesGrid');
                    if (grid) {
                        grid.innerHTML = '<p>Search failed. Please try again.</p>';
                    }
                }
            }
        }

        // Initialize group members array
        window.groupMembers = [];
        
        // Group member management functions
        function addGroupMember(memberData) {
            if (!window.groupMembers) {
                window.groupMembers = [];
            }
            window.groupMembers.push(memberData);
            console.log('Added member:', memberData);
            console.log('Total members:', window.groupMembers.length);
        }
        
        function saveMemberFromPage(formData) {
            const member = {
                name: formData.get('name'),
                age: calculateAge(formData.get('date_of_birth')),
                ability: parseInt(formData.get('physical_ability')) || 5,
                interests: formData.getAll('interests')
            };
            addGroupMember(member);
            alert('Member added successfully!');
        }
        
        // Event listeners
        document.addEventListener('DOMContentLoaded', function() {
            // Navigation
            logoutBtn.addEventListener('click', logout);



            // Auth tabs
            document.querySelectorAll('.auth-tab').forEach(tab => {
                tab.addEventListener('click', (e) => {
                    e.preventDefault();
                    switchAuthTab(tab.dataset.tab);
                });
            });

            // Auth forms
            document.getElementById('loginForm').addEventListener('submit', (e) => {
                e.preventDefault();
                const email = document.getElementById('loginEmail').value;
                const password = document.getElementById('loginPassword').value;
                
                // Simple validation for demo
                if (email && password) {
                    login();
                } else {
                    alert('Please fill in all fields');
                }
            });
            
            document.getElementById('registerForm').addEventListener('submit', (e) => {
                e.preventDefault();
                const name = document.getElementById('registerName').value;
                const email = document.getElementById('registerEmail').value;
                const password = document.getElementById('registerPassword').value;
                const confirmPassword = document.getElementById('confirmPassword').value;
                const agreeTerms = document.getElementById('agreeTerms').checked;
                
                // Simple validation for demo
                if (!name || !email || !password || !confirmPassword) {
                    alert('Please fill in all fields');
                    return;
                }
                
                if (password !== confirmPassword) {
                    alert('Passwords do not match');
                    return;
                }
                
                if (!agreeTerms) {
                    alert('Please agree to the terms and conditions');
                    return;
                }

                login();
            });

            // Initial switch auth mode listener
            document.getElementById('switchAuthMode').addEventListener('click', (e) => {
                e.preventDefault();
                switchAuthTab('register');
            });

            // Search inputs
            document.getElementById('searchInput').addEventListener('input', (e) => {
                performSearch(e.target.value);
            });
            
            document.getElementById('mainSearchInput').addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    showView('search');
                    document.getElementById('searchInput').value = e.target.value;
                    performSearch(e.target.value);
                }
            });

            // Filter controls
            document.getElementById('minAge').addEventListener('input', (e) => {
                document.getElementById('minAgeValue').textContent = e.target.value;
                performSearch(document.getElementById('searchInput').value);
            });
            
            document.getElementById('maxAge').addEventListener('input', (e) => {
                document.getElementById('maxAgeValue').textContent = e.target.value;
                performSearch(document.getElementById('searchInput').value);
            });

            // Filter dropdowns
            ['durationFilter', 'costFilter', 'seasonFilter', 'playersFilter'].forEach(id => {
                document.getElementById(id).addEventListener('change', () => {
                    performSearch(document.getElementById('searchInput').value);
                });
            });

            // Toggle buttons
            document.querySelectorAll('.toggle-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const filter = e.target.dataset.filter;
                    const value = e.target.dataset.value;
                    
                    // Remove active from siblings
                    document.querySelectorAll(`[data-filter="${filter}"]`).forEach(sibling => {
                        sibling.classList.remove('active');
                    });
                    
                    // Add active to clicked button
                    e.target.classList.add('active');
                    
                    performSearch(document.getElementById('searchInput').value);
                });
            });

            // Add activity form
            document.getElementById('addActivityForm').addEventListener('submit', (e) => {
                e.preventDefault();
                const formData = new FormData(e.target);
                
                const newActivity = {
                    title: formData.get('title'),
                    age_min: parseInt(formData.get('age_min')),
                    age_max: parseInt(formData.get('age_max')),
                    duration_mins: parseInt(formData.get('duration')),
                    cost: formData.get('cost'),
                    indoor_outdoor: formData.get('indoor_outdoor'),
                    season: formData.getAll('season'),
                    players: formData.get('players') || '1+',
                    tags: formData.get('tags') ? formData.get('tags').split(',').map(t => t.trim()) : [],
                    materials_needed: formData.get('materials').split(',').map(m => m.trim()),
                    how_to_play: formData.get('instructions'),
                    parent_caution: formData.get('caution') || ''
                };
                
                // Add to activities list
                sampleActivities.push(newActivity);

                e.target.reset();
            });

            // Save draft button
            document.getElementById('saveDraftBtn').addEventListener('click', () => {
                // Draft saved silently
            });


            
            // Member page form submission
            document.getElementById('memberPageForm').addEventListener('submit', (e) => {
                e.preventDefault();
                const formData = new FormData(e.target);
                
                // Validation
                if (!formData.get('name') || !formData.get('date_of_birth') || 
                    !formData.get('height_cm') || !formData.get('weight_kg')) {
                    alert('Please fill in all required fields (marked with *)');
                    return;
                }
                
                const height = parseInt(formData.get('height_cm'));
                const weight = parseFloat(formData.get('weight_kg'));
                const birthDate = new Date(formData.get('date_of_birth'));
                const today = new Date();
                
                if (height < 50 || height > 250) {
                    alert('Height must be between 50-250 cm');
                    return;
                }
                
                if (weight < 5 || weight > 200) {
                    alert('Weight must be between 5-200 kg');
                    return;
                }
                
                if (birthDate > today) {
                    alert('Date of birth cannot be in the future');
                    return;
                }
                
                const age = calculateAge(formData.get('date_of_birth'));
                if (age > 100) {
                    alert('Age must be between 0-100');
                    return;
                }
                
                saveMemberFromPage(formData);
            });
            
            // BMI auto-calculation for member page form
            function updateBMIPage() {
                const height = document.querySelector('#memberPageForm input[name="height_cm"]')?.value;
                const weight = document.querySelector('#memberPageForm input[name="weight_kg"]')?.value;
                const bmiField = document.querySelector('#memberPageForm input[name="bmi"]');
                
                if (height && weight && bmiField) {
                    const bmi = calculateBMI(parseInt(height), parseFloat(weight));
                    bmiField.value = bmi;
                }
            }            

            // Range slider value display updates - for main view ranges
            function updateRangeDisplay(rangeInput, displayElement = null) {
                const value = rangeInput.value;
                const label = rangeInput.previousElementSibling;
                if (label && label.classList.contains('form-label')) {
                    const baseText = label.textContent.split('(')[0].trim();
                    label.textContent = `${baseText} (${value})`;
                }
            }
            
            // Add range input listeners for search filters
            document.querySelectorAll('#search input[type="range"]').forEach(range => {
                range.addEventListener('input', (e) => {
                    updateRangeDisplay(e.target);
                });
                
                // Initialize display
                updateRangeDisplay(range);
            });
            
            // Setup range listeners for member form when it becomes active
            function setupMemberFormRangeListeners() {
                const memberForm = document.getElementById('member-form');
                if (memberForm) {
                    memberForm.querySelectorAll('input[type="range"]').forEach(range => {
                        // Remove existing listeners to avoid duplicates
                        range.removeEventListener('input', updateMemberFormRangeDisplays);
                        range.addEventListener('input', (e) => {
                            const value = e.target.value;
                            const label = e.target.previousElementSibling;
                            if (label && label.classList.contains('form-label')) {
                                const baseText = label.textContent.split('(')[0].trim();
                                label.textContent = `${baseText} (${value})`;
                            }
                        });
                        
                        // Initialize display
                        const value = range.value;
                        const label = range.previousElementSibling;
                        if (label && label.classList.contains('form-label')) {
                            const baseText = label.textContent.split('(')[0].trim();
                            label.textContent = `${baseText} (${value})`;
                        }
                    });
                }
            }
            
            // Get Recommendations button
            document.getElementById('getRecommendationsBtn').addEventListener('click', getPersonalizedRecommendations);

            // Guest mode tooltip functionality
            const guestBadge = document.getElementById('guestBadge');
            const guestTooltip = document.getElementById('guestTooltip');
            
            if (guestBadge && guestTooltip) {
                guestBadge.addEventListener('mouseenter', () => {
                    guestTooltip.classList.remove('hidden');
                });
                
                guestBadge.addEventListener('mouseleave', () => {
                    guestTooltip.classList.add('hidden');
                });
            }
            
            // Auto-start in guest mode on first load
            startGuestMode();

            // Load activities from CSV and initialize display
            loadActivitiesFromCSV();

            // Initialize range displays
            setTimeout(() => {
                document.querySelectorAll('input[type="range"]').forEach(range => {
                    updateRangeDisplay(range);
                });
            }, 100);
        });
        
        // Global functions for onclick handlers
        window.editMemberPage = editMemberPage;
        window.duplicateMember = duplicateMember;
        window.removeMember = removeMember;
        window.addMemberTemplate = addMemberTemplate;
        // Note: getPersonalizedRecommendations is defined in api-integration.js
        window.setupMemberPageForm = setupMemberPageForm;
        window.addToPlan = addToPlan;
        window.exportToICS = exportToICS;
        window.exportToCSV = exportToCSV;
        window.exportToPDF = exportToPDF;
        
        // Setup BMI calculation for member form when view is shown
        function setupMemberFormEventListeners() {
            const heightInput = document.querySelector('#member-form input[name="height_cm"]');
            const weightInput = document.querySelector('#member-form input[name="weight_kg"]');
            
            if (heightInput && weightInput) {
                heightInput.addEventListener('input', updateBMIPage);
                weightInput.addEventListener('input', updateBMIPage);
            }
            
            // Setup range sliders
            setupMemberFormRangeListeners();
        }

        // My Group indicator functions
        let useMyGroupFilter = true;

        function updateMyGroupIndicator() {
            const indicator = document.getElementById('myGroupIndicator');
            if (!indicator) return;

            const memberCount = document.getElementById('groupMemberCount');
            const ageRange = document.getElementById('groupAgeRange');

            if (groupMembers.length > 0 && useMyGroupFilter) {
                const ages = groupMembers.map(m => m.age);
                const minAge = Math.min(...ages);
                const maxAge = Math.max(...ages);

                memberCount.textContent = groupMembers.length;
                ageRange.textContent = minAge === maxAge ? `${minAge}` : `${minAge}-${maxAge}`;
                indicator.classList.remove('hidden');
            } else {
                indicator.classList.add('hidden');
            }
        }

        function toggleMyGroupFilter() {
            useMyGroupFilter = !useMyGroupFilter;
            const toggleText = document.getElementById('myGroupToggleText');
            if (toggleText) {
                toggleText.textContent = useMyGroupFilter ? 'Disable' : 'Enable';
            }

            updateMyGroupIndicator();

            // Re-run search if there's a query
            const searchInput = document.getElementById('searchInput');
            if (searchInput && searchInput.value) {
                performSearch(searchInput.value);
            }
        }

        // Make functions globally accessible
        window.toggleMyGroupFilter = toggleMyGroupFilter;
        window.updateMyGroupIndicator = updateMyGroupIndicator;
