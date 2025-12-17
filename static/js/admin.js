document.addEventListener('DOMContentLoaded', () => {
    const usersTbody = document.getElementById('users-tbody');
    const loadingDiv = document.getElementById('loading');
    const usersTable = document.getElementById('users-table');
    const searchInput = document.getElementById('search-input');
    const paginationControls = document.getElementById('pagination-controls');
    const prevPageBtn = document.getElementById('prev-page');
    const nextPageBtn = document.getElementById('next-page');
    const pageInfoSpan = document.getElementById('page-info');

    let allUsers = [];
    let filteredUsers = [];
    let currentPage = 1;
    const rowsPerPage = 10;

    // --- Fonctions de rendu ---

    function renderTable(users) {
        usersTbody.innerHTML = '';
        if (users.length === 0) {
            showNoResults();
            return;
        }

        const startIndex = (currentPage - 1) * rowsPerPage;
        const endIndex = startIndex + rowsPerPage;
        const paginatedUsers = users.slice(startIndex, endIndex);

        paginatedUsers.forEach(user => {
            const row = document.createElement('tr');
            row.dataset.userId = user.user_id;

            const statusClass = user.status === 'active' ? 'status-active' : 'status-deleted';
            const actionButton = user.status === 'active'
                ? `<button class="btn-delete" onclick="handleAction('${user.user_id}', 'delete')">Désactiver</button>`
                : `<button class="btn-reactivate" onclick="handleAction('${user.user_id}', 'reactivate')">Réactiver</button>`;

            row.innerHTML = `
                <td>${user.user_id}</td>
                <td><span class="status ${statusClass}">${user.status === 'active' ? 'Actif' : 'Supprimé'}</span></td>
                <td>${actionButton}</td>
            `;
            usersTbody.appendChild(row);
        });
    }

    function showNoResults() {
        const row = document.createElement('tr');
        row.innerHTML = `<td colspan="3" style="text-align: center; font-style: italic;">Aucun utilisateur trouvé.</td>`;
        usersTbody.appendChild(row);
    }

    function updatePagination(users) {
        const totalPages = Math.ceil(users.length / rowsPerPage);
        if (totalPages <= 1) {
            paginationControls.style.display = 'none';
            return;
        }

        paginationControls.style.display = 'flex';
        pageInfoSpan.textContent = `Page ${currentPage} sur ${totalPages}`;
        prevPageBtn.disabled = currentPage === 1;
        nextPageBtn.disabled = currentPage === totalPages;
    }

    // --- Fonctions de gestion des données ---

    async function fetchUsers() {
        try {
            // Simuler un appel API pour l'exemple
            // Dans un cas réel, ce serait : const response = await fetch('/api/users');
            // const data = await response.json();
            // allUsers = data.users;
            
            // Données de simulation
            allUsers = Array.from({ length: 53 }, (_, i) => ({
                user_id: 1000 + i,
                status: Math.random() > 0.2 ? 'active' : 'deleted'
            }));

            filteredUsers = [...allUsers];
            displayData();
        } catch (error) {
            console.error("Erreur lors de la récupération des utilisateurs:", error);
            loadingDiv.textContent = "Erreur lors du chargement des données.";
        } finally {
            loadingDiv.style.display = 'none';
            usersTable.style.display = 'table';
        }
    }

    function filterUsers() {
        const searchTerm = searchInput.value.toLowerCase();
        filteredUsers = allUsers.filter(user => 
            user.user_id.toString().includes(searchTerm)
        );
        currentPage = 1; // Reset to first page after search
        displayData();
    }

    function displayData() {
        renderTable(filteredUsers);
        updatePagination(filteredUsers);
    }

    // --- Gestionnaires d'événements ---

    searchInput.addEventListener('input', filterUsers);

    prevPageBtn.addEventListener('click', () => {
        if (currentPage > 1) {
            currentPage--;
            displayData();
        }
    });

    nextPageBtn.addEventListener('click', () => {
        const totalPages = Math.ceil(filteredUsers.length / rowsPerPage);
        if (currentPage < totalPages) {
            currentPage++;
            displayData();
        }
    });

    // --- Actions (simulées) ---
    window.handleAction = (userId, action) => {
        console.log(`Action: ${action} sur l'utilisateur ${userId}`);
        // Ici, vous appelleriez votre API pour effectuer l'action
        // Puis vous mettriez à jour l'état local et ré-afficherez les données
        const user = allUsers.find(u => u.user_id == userId);
        if (user) {
            user.status = action === 'delete' ? 'deleted' : 'active';
            filterUsers(); // Re-render the table with the new state
        }
    };

    // --- Initialisation ---
    fetchUsers();
});