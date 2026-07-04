const menu = document.querySelector('.menu');
const nav = document.querySelector('.nav-links');
menu?.addEventListener('click', () => {
  const open = nav.classList.toggle('open');
  menu.setAttribute('aria-expanded', String(open));
});

document.querySelectorAll('[data-repo-link]').forEach(a => {
  a.addEventListener('click', e => {
    if (a.getAttribute('href') === '#') {
      e.preventDefault();
      alert('Replace the placeholder GitHub link in index.html with your repository URL.');
    }
  });
});

const copy = document.querySelector('#copy-citation');
copy?.addEventListener('click', async () => {
  const text = document.querySelector('.code-card code').innerText;
  await navigator.clipboard.writeText(text);
  copy.textContent = 'Copied';
  setTimeout(() => copy.textContent = 'Copy', 1400);
});
