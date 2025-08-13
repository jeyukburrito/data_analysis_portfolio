document.addEventListener('DOMContentLoaded', function () {

    // Initialize Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Add hover effect and tooltip to achievement cards
    const achievementCards = document.querySelectorAll('.achievement-card');

    achievementCards.forEach(card => {
        const title = card.querySelector('h5').innerText;
        const description = card.querySelector('p').innerText;
        let status = '잠김';

        if (card.classList.contains('unlocked')) {
            status = '획득 완료';
        }

        card.setAttribute('data-bs-toggle', 'tooltip');
        card.setAttribute('data-bs-placement', 'top');
        card.setAttribute('title', `${title}: ${description} (${status})`);
    });

    // Re-initialize tooltips after dynamically adding attributes
    const newTooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    newTooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

});
