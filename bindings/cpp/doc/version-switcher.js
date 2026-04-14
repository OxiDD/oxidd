// spell-checker:ignore projectnumber
document.addEventListener("DOMContentLoaded", async () => {
  let versions = [];
  try {
    const response = await fetch("https://oxidd.net/api/cpp/versions.json");
    if (!response.ok) return;
    versions = await response.json();
  } catch {
    return;
  }

  const switcherSpan = document.getElementById("projectnumber");
  const currentVer = switcherSpan.innerText.trim();
  const dropdown = document.createElement("ul");
  dropdown.classList.add("dropdown");

  const host = window.location.host;
  const path = window.location.pathname.replace("/stable/", `/v${currentVer}/`);
  let page = path.substring(path.lastIndexOf("/") + 1);
  if (page == "index.html") page = "";

  for (const version of versions) {
    const a = document.createElement("a");
    let baseURL = version.url;
    if (!baseURL.endsWith("/")) baseURL += "/";
    a.dataset.hrefPage = baseURL + page;
    a.href = a.dataset.hrefPage + window.location.hash;
    a.appendChild(document.createTextNode(version.name ?? version.version));

    const li = document.createElement("li");
    li.appendChild(a);
    dropdown.appendChild(li);

    try {
      const url = new URL(version.url);
      if (url.host == host && path.startsWith(url.pathname))
        a.classList.add("active");
    } catch {}
  }

  window.addEventListener("hashchange", () => {
    for (const li of dropdown.children) {
      const a = li.firstElementChild;
      a.href = a.dataset.hrefPage + window.location.hash;
    }
  });

  const dropdownIndicator = document.createElement("span");
  dropdownIndicator.appendChild(document.createTextNode("⏷"));
  dropdownIndicator.classList.add("dropdown-indicator");
  switcherSpan.appendChild(dropdownIndicator);
  switcherSpan.appendChild(dropdown);
});
