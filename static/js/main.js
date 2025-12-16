document.addEventListener('DOMContentLoaded', function() {
    const userInput = document.getElementById('user-input');
    const createUserButton = document.getElementById('create-user-button');
    const filterCountrySelect = document.getElementById('filter-country');
    const filterDeviceSelect = document.getElementById('filter-device');

    const articleTitleInput = document.getElementById('article-title');
    const articleContentInput = document.getElementById('article-content');
    const articleCategoryInput = document.getElementById('article-category');
    const userContextDisplay = document.getElementById('user-context-display');
    const contextCountrySpan = document.getElementById('context-country');
    const contextDeviceSpan = document.getElementById('context-device');
    const deleteUserButton = document.getElementById('delete-user-button');
    const addArticleButton = document.getElementById('add-article-button');
    const mainLoader = document.getElementById('main-loader');

    const mainContentArea = document.getElementById('main-content-area');
    // New tab-related constants
    const tabButtons = document.querySelectorAll('.tab-button');
    const recommendationsTabContent = document.getElementById('recommendations-tab-content');
    const historyTabContent = document.getElementById('history-tab-content');
    const globalTrendsTabContent = document.getElementById('global-trends-tab-content');
    const performanceTabContent = document.getElementById('performance-tab-content');
    const initialRecoMessage = document.getElementById('initial-reco-message');

    // Modal for adding articles
    const addArticleModal = document.getElementById('add-article-modal');
    const openAddArticleModalBtn = document.getElementById('open-add-article-modal-btn');
    const closeAddArticleModalBtn = document.getElementById('close-add-article-modal-btn');

    let debounceTimer;
    let currentActiveTab = 'recommendations'; // Keep track of the currently active tab

    // --- Notification System ---
    const notificationContainer = document.getElementById('notification-container');
    async function showNotification(message, type = 'info') { // type can be 'info', 'success', 'error'
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;

        // Si le message est un objet Réponse, essayer d'extraire l'erreur JSON.
        if (message instanceof Response) {
            const errorData = await message.json().catch(() => ({ error: "Erreur de décodage de la réponse." }));
            notification.textContent = `Erreur ${message.status}: ${errorData.error || message.statusText}`;
        } else {
            notification.textContent = message;
        }

        notificationContainer.appendChild(notification);
        setTimeout(() => {
            notification.remove();
        }, 5000); // Auto-dismiss after 5 seconds
    }

    // Fonctions pour gérer l'indicateur de chargement
    function showLoader() {
        mainLoader.classList.remove('hidden');
        const activeTabContent = document.querySelector('.tab-content.active');
        if (activeTabContent) {
            activeTabContent.innerHTML = '';
        }
    }

    function hideLoader() {
        mainLoader.classList.add('hidden');
    }

    // Fonction pour peupler les filtres de pays et d'appareil
    function populateFilters() {
        fetch('/api/global_trends') // Réutilise l'API des tendances pour obtenir les options de filtre
            .then(response => response.json())
            .then(data => {
                filterCountrySelect.innerHTML = '<option value="">Tous</option>';
                data.clicks_by_country.forEach(item => {
                    const option = document.createElement('option');
                    option.value = item.country;
                    option.textContent = item.country;
                    filterCountrySelect.appendChild(option);
                });

                filterDeviceSelect.innerHTML = '<option value="">Tous</option>';
                data.clicks_by_device.forEach(item => {
                    const option = document.createElement('option');
                    option.value = item.deviceGroup;
                    option.textContent = item.deviceGroup;
                    filterDeviceSelect.appendChild(option);
                });
            })
            .catch(error => console.error("Erreur lors du chargement des options de filtre:", error));
    }
    // Fonction pour charger et afficher le contexte de l'utilisateur
    function loadUserContext(userId) {
        if (!userId) {
            userContextDisplay.style.display = 'none';
            return;
        }
        fetch(`/api/user_context/${userId}`)
            .then(response => response.json())
            .then(context => {
                if (context.country && context.deviceGroup) {
                    contextCountrySpan.textContent = context.country;
                    contextDeviceSpan.textContent = context.deviceGroup;
                    userContextDisplay.style.display = 'block';
                } else {
                    userContextDisplay.style.display = 'none';
                }
            })
            .catch(error => {
                console.error("Erreur lors du chargement du contexte utilisateur:", error);
                userContextDisplay.style.display = 'none';
                contextCountrySpan.textContent = 'N/A';
                contextDeviceSpan.textContent = 'N/A';
            });
    }

    // Fonction pour valider l'ID utilisateur
    function isValidUserId(id) {
        const num = Number(id);
        // Vérifie si c'est un nombre, un entier, et s'il est positif.
        return !isNaN(num) && Number.isInteger(num) && num > 0;
    }

    // Gérer le clic sur le bouton de création d'utilisateur
    createUserButton.addEventListener('click', function() {
        createUserButton.textContent = 'Création en cours...';
        createUserButton.disabled = true;

        // On s'assure que le bouton de suppression est caché pendant la création
        deleteUserButton.style.display = 'none';
        fetch('/api/users', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if (data.user_id) {
                    showNotification(`Nouvel utilisateur créé avec l'ID : ${data.user_id}`, 'success');
                    userInput.value = data.user_id; // Set the new user ID in the input
                    loadUserContext(data.user_id); // Load context for the new user
                    deleteUserButton.style.display = 'block'; // Afficher le bouton de suppression pour le nouvel utilisateur
                }
            })
            .catch(error => {
                console.error("Erreur lors de la création de l'utilisateur:", error); // NOSONAR
                showNotification("Impossible de créer un nouvel utilisateur. Vérifiez la console pour les détails.", 'error');
            })
            .finally(() => {
                createUserButton.textContent = 'Créer un nouvel utilisateur';
                createUserButton.disabled = false;
            });
    });

    // Écouter le changement dans l'entrée utilisateur
    userInput.addEventListener('input', function() {
        // Annuler le minuteur précédent pour éviter les appels multiples
        clearTimeout(debounceTimer);

        // Démarrer un nouveau minuteur
        debounceTimer = setTimeout(() => {
            const selectedUserId = userInput.value;
            loadUserContext(selectedUserId);
            // Afficher ou cacher le bouton de suppression
            if (isValidUserId(selectedUserId)) {
                deleteUserButton.style.display = 'block';
            } else {
                deleteUserButton.style.display = 'none';
            }
            // If recommendations tab is active, refresh recommendations
            if (currentActiveTab === 'recommendations') {
                displayRecommendations(selectedUserId, filterCountrySelect.value, filterDeviceSelect.value);
            }
        }, 400); // Attendre 400ms après la dernière frappe avant d'exécuter
    });

    // Gérer le clic sur le bouton de suppression d'utilisateur
    deleteUserButton.addEventListener('click', function() {
        const userId = userInput.value;
        if (!isValidUserId(userId)) return;

        if (confirm(`Êtes-vous sûr de vouloir désactiver l'utilisateur ${userId} ? Il n'apparaîtra plus dans la liste, mais ses données seront conservées.`)) { // NOSONAR
            fetch(`/api/users/${userId}`, { method: 'DELETE' })
                .then(response => {
                    if (!response.ok) throw response; // Lancer la réponse pour la capture d'erreur
                    return response.json();
                })
                .then(data => {
                    showNotification(data.message, 'success');
                    userInput.value = ''; // Vider le champ de saisie
                    loadUserContext(''); // Clear user context display
                    switchTab('recommendations'); // Go back to recommendations tab, which will show initial message
                    deleteUserButton.style.display = 'none'; // Cacher le bouton
                });
        }
    });

    // Helper function to safely create a recommendation card element
    function createRecoCard(reco) {
        const card = document.createElement('div');
        card.className = 'reco-card';
        card.dataset.articleId = reco.article_id;

        const title = document.createElement('h3');
        title.textContent = reco.title || 'Titre non disponible';

        const content = document.createElement('p');
        content.textContent = (reco.content || 'Contenu non disponible').substring(0, 200) + '...';

        const controls = document.createElement('div');
        controls.className = 'rating-controls';
        controls.innerHTML = `
            <select class="rating-select">
                <option value="1">⭐</option>
                <option value="2">⭐⭐</option>
                <option value="3" selected>⭐⭐⭐</option>
                <option value="4">⭐⭐⭐⭐</option>
                <option value="5">⭐⭐⭐⭐⭐</option>
            </select>
            <button class="rate-button">Noter</button>
        `;

        card.append(title, content, controls);
        return card;
    }

    // Gérer la soumission d'une note (délégation d'événement)
    mainContentArea.addEventListener('click', function(event) {
        if (event.target.classList.contains('rate-button')) {
            const card = event.target.closest('.reco-card');
            const articleId = card.dataset.articleId;
            const rating = card.querySelector('.rating-select').value; // NOSONAR
            const userId = userInput.value;

            if (!isValidUserId(userId)) {
                showNotification("Veuillez entrer un ID utilisateur valide pour noter un article.", 'error');
                return;
            }

            fetch('/api/interactions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_id: parseInt(userId),
                    article_id: parseInt(articleId),
                    rating: parseInt(rating)
                })
            })
            .then(response => {
                if (response.ok) {
                    event.target.textContent = 'Merci !';
                    event.target.closest('.reco-card').querySelector('.rating-select').disabled = true;
                } else {
                    showNotification('Erreur lors de la notation.', 'error');
                }
            });
        }
    });

    // New function to display recommendations
    function displayRecommendations(userId, country = '', device = '') {
        recommendationsTabContent.innerHTML = ''; // Clear previous recommendations
        if (!isValidUserId(userId)) {
            initialRecoMessage.classList.remove('hidden'); // Show initial message if no valid user
            recommendationsTabContent.appendChild(initialRecoMessage);
            return;
        }
        initialRecoMessage.classList.add('hidden'); // Hide initial message

        showLoader();
        const queryParams = new URLSearchParams({ user_id: userId });
        if (country) { queryParams.append('country', country); }
        if (device) { queryParams.append('device', device); }

        fetch(`/api/recommendations?${queryParams.toString()}`)
            .then(response => {
                if (!response.ok) throw response;
                return response.json();
            })
            .then(recos => {
                hideLoader();
                if (Array.isArray(recos) && recos.length > 0) {
                    recos.forEach(reco => {
                        recommendationsTabContent.appendChild(createRecoCard(reco));
                    });
                } else {
                    recommendationsTabContent.innerHTML = '<p>Aucune recommandation trouvée pour cet utilisateur avec les filtres actuels.</p>';
                }
            })
            .catch(error => {
                hideLoader();
                showNotification(error, 'error');
                recommendationsTabContent.innerHTML = '<p class="error-message">Impossible de charger les recommandations. Vérifiez que l\'identité managée a le rôle "Storage Blob Data Reader" sur le compte de stockage.</p>';
            });
    }

    // New function to display history
    function displayHistory(userId) {
        if (!isValidUserId(userId)) {
            showNotification('Veuillez entrer un ID utilisateur valide pour voir l\'historique.', 'error');
            return;
        }
        showLoader();
        fetch(`/api/history/${userId}`)
            .then(response => response.json())
            .then(history => {
                hideLoader();
                historyTabContent.innerHTML = '<h2>Votre Historique de Notation</h2>';
                if (Array.isArray(history) && history.length > 0) {
                    history.forEach(item => {
                        const date = item.click_timestamp ? new Date(item.click_timestamp * 1000).toLocaleString('fr-FR') : 'Date inconnue';
                        historyTabContent.innerHTML += `
                            <div class="reco-card">
                                <h3>${item.title || 'Titre non disponible'}</h3>
                                <p><strong>Votre note : ${item.nb}</strong> | Noté le : ${date}</p>
                            </div>`;
                    });
                } else {
                    historyTabContent.innerHTML += '<p>Vous n\'avez encore noté aucun article.</p>';
                }
            })
            .catch(error => {
                hideLoader();
                console.error("Erreur lors du chargement de l'historique:", error);
                historyTabContent.innerHTML = '<p>Impossible de charger l\'historique.</p>';
            });
    }

    // Gérer le clic sur le bouton d'ajout d'article
    addArticleButton.addEventListener('click', function() {
        const title = articleTitleInput.value;
        const content = articleContentInput.value;
        const categoryId = articleCategoryInput.value;

        if (!title || !content || !categoryId) {
            showNotification("Veuillez remplir tous les champs pour ajouter un article.", 'error');
            return;
        }

        // --- Amélioration de l'UX : Ajout d'un spinner et gestion des erreurs ---
        addArticleButton.textContent = 'Ajout en cours...';
        addArticleButton.disabled = true;
        // Remplacer le texte par un spinner (nécessite CSS pour .btn-spinner)
        addArticleButton.innerHTML = '<span class="btn-spinner"></span> Ajout en cours...';

        fetch('/api/articles', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title, content, category_id: parseInt(categoryId) })
        })
        .then(response => {
            if (!response.ok) {
                // Si l'erreur est une erreur d'authentification, afficher un message spécifique.
                if (response.status === 401) {
                    throw new Error("Action non autorisée. Veuillez vous connecter en tant qu'administrateur.");
                }
                throw new Error(`Erreur HTTP ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            showNotification(`Article ajouté avec l'ID : ${data.article_id}`, 'success');
            articleTitleInput.value = ''; // Réinitialiser le champ
            articleContentInput.value = ''; // Réinitialiser le champ
            addArticleModal.classList.remove('visible'); // Fermer la modale après succès
        })
        .catch(error => {
            console.error("Erreur lors de l'ajout de l'article:", error);
            showNotification(error.message, 'error');
        })
        .finally(() => {
            addArticleButton.innerHTML = "Ajouter l'article"; // Restaurer le texte original
            addArticleButton.disabled = false;
        });
    });
    
    function displayGlobalTrends() {
        showLoader();
        fetch('/api/global_trends')
            .then(response => response.json())
            .then(data => {
                hideLoader();
                globalTrendsTabContent.innerHTML = `
                    <h2>Tendances Globales des Clics</h2>
                    <div style="display: flex; flex-wrap: wrap; gap: 20px;">
                        <div style="flex: 1; min-width: 300px;">
                            <h3>Clics par Pays</h3>
                            <canvas id="country-chart"></canvas>
                        </div>
                        <div style="flex: 1; min-width: 300px;">
                            <h3>Clics par Type d'Appareil</h3>
                            <canvas id="device-chart"></canvas>
                        </div>
                    </div>
                `;

                // Graphique par Pays
                const countryCtx = document.getElementById('country-chart').getContext('2d');
                new Chart(countryCtx, {
                    type: 'pie', // Ou 'bar'
                    data: {
                        labels: data.clicks_by_country.map(item => item.country),
                        datasets: [{
                            label: 'Nombre de Clics',
                            data: data.clicks_by_country.map(item => item.count),
                            backgroundColor: [
                                '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40', '#E7E9ED'
                            ],
                            hoverOffset: 4
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            title: {
                                display: true,
                                text: 'Distribution des clics par pays'
                            }
                        }
                    }
                });

                // Graphique par Type d'Appareil
                const deviceCtx = document.getElementById('device-chart').getContext('2d');
                new Chart(deviceCtx, {
                    type: 'bar', // Ou 'pie'
                    data: {
                        labels: data.clicks_by_device.map(item => item.deviceGroup),
                        datasets: [{
                            label: 'Nombre de Clics',
                            data: data.clicks_by_device.map(item => item.count),
                            backgroundColor: [
                                '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0'
                            ]
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            title: {
                                display: true,
                                text: 'Distribution des clics par type d\'appareil'
                            }
                        }
                    }
                });
            })
            .catch(error => {
                hideLoader();
                console.error("Erreur lors du chargement des tendances globales:", error);
                globalTrendsTabContent.innerHTML = '<p>Impossible de charger les tendances globales.</p>';
            });
    }

    function displayModelPerformance() {
        showLoader();
        fetch('/api/performance')
            .then(response => response.json())
            .then(data => {
                hideLoader();
                if (!Array.isArray(data) || data.length === 0) {
                    performanceTabContent.innerHTML = '<p>Aucune donnée de performance du modèle trouvée.</p>';
                    return;
                }

                performanceTabContent.innerHTML = `
                    <h2>Évolution des Performances du Modèle</h2>
                    <canvas id="performance-chart"></canvas>
                `;

                const perfCtx = document.getElementById('performance-chart').getContext('2d');
                const labels = data.map(d => new Date(d.timestamp).toLocaleDateString('fr-FR'));

                new Chart(perfCtx, {
                    type: 'line',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: 'Précision@10 (Validation)',
                            data: data.map(d => d.precision_at_10),
                            borderColor: 'rgb(75, 192, 192)',
                            backgroundColor: 'rgba(75, 192, 192, 0.2)',
                            fill: true,
                            tension: 0.1
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            title: {
                                display: true,
                                text: 'Évolution de la Précision@10 par entraînement'
                            }
                        }
                    }
                });
            })
            .catch(error => {
                hideLoader();
                console.error("Erreur lors du chargement des performances:", error);
                performanceTabContent.innerHTML = '<p>Impossible de charger les données de performance.</p>';
            });
    }

    // Function to switch between tabs
    function switchTab(tabName) {
        // Hide all tab contents
        recommendationsTabContent.classList.remove('active');
        historyTabContent.classList.remove('active');
        globalTrendsTabContent.classList.remove('active');
        performanceTabContent.classList.remove('active');

        // Deactivate all tab buttons
        tabButtons.forEach(button => button.classList.remove('active'));

        // Show the selected tab content and activate its button
        let userId = userInput.value;
        let country = filterCountrySelect.value;
        let device = filterDeviceSelect.value;

        switch (tabName) {
            case 'recommendations':
                recommendationsTabContent.classList.add('active');
                document.querySelector('.tab-button[data-tab="recommendations"]').classList.add('active');
                displayRecommendations(userId, country, device);
                break;
            case 'history':
                historyTabContent.classList.add('active');
                document.querySelector('.tab-button[data-tab="history"]').classList.add('active');
                displayHistory(userId);
                break;
            case 'global-trends':
                globalTrendsTabContent.classList.add('active');
                document.querySelector('.tab-button[data-tab="global-trends"]').classList.add('active');
                displayGlobalTrends();
                break;
            case 'performance':
                performanceTabContent.classList.add('active');
                document.querySelector('.tab-button[data-tab="performance"]').classList.add('active');
                displayModelPerformance();
                break;
        }
        currentActiveTab = tabName; // Update the active tab tracker
    }

    // Event listener for tab buttons
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            switchTab(button.dataset.tab);
        });
    });

    // --- Modal Logic for Adding Articles ---
    openAddArticleModalBtn.addEventListener('click', () => addArticleModal.classList.add('visible'));
    closeAddArticleModalBtn.addEventListener('click', () => addArticleModal.classList.remove('visible'));
    addArticleModal.addEventListener('click', (e) => {
        if (e.target === addArticleModal) addArticleModal.classList.remove('visible');
    });

    // Event listeners for filter changes (should only affect recommendations tab)
    filterCountrySelect.addEventListener('change', () => {
        if (currentActiveTab === 'recommendations') { displayRecommendations(userInput.value, filterCountrySelect.value, filterDeviceSelect.value); }
    });
    filterDeviceSelect.addEventListener('change', () => {
        if (currentActiveTab === 'recommendations') { displayRecommendations(userInput.value, filterCountrySelect.value, filterDeviceSelect.value); }
    });

    // Fonction pour vérifier et afficher le statut du réentraînement
    function checkRetrainingStatus() {
        const statusIndicator = document.getElementById('retraining-status-indicator');
        const dot = statusIndicator.querySelector('.dot');
        const text = statusIndicator.querySelector('.status-text');

        fetch('/api/retraining_status')
            .then(response => response.json())
            .then(data => {
                dot.className = 'dot'; // reset
                switch(data.status) {
                    case 'in_progress':
                        dot.classList.add('in-progress');
                        text.textContent = 'Réentraînement en cours...';
                        break;
                    case 'failed':
                        dot.classList.add('failed');
                        text.textContent = 'Échec du réentraînement';
                        break;
                    case 'idle':
                    default:
                        dot.classList.add('idle');
                        text.textContent = 'Actif';
                        break;
                }
            })
            .catch(() => { text.textContent = 'Statut inconnu'; });
    }

    // Chargement initial des utilisateurs
    populateFilters();
    // Initial load of user context if an ID is pre-filled (e.g., from browser history)
    loadUserContext(userInput.value); // This will also trigger recommendations if the tab is active
    checkRetrainingStatus(); // Vérifier le statut au chargement
    setInterval(checkRetrainingStatus, 30000); // Puis toutes les 30 secondes

    // Activate the default tab on initial load
    switchTab('recommendations');
});