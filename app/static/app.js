const form = document.querySelector("#queryForm");
const input = document.querySelector("#queryInput");
const searchButton = document.querySelector("#searchButton");
const clearButton = document.querySelector("#clearCache");
const results = document.querySelector("#results");
const apiDot = document.querySelector("#apiDot");
const apiStatus = document.querySelector("#apiStatus");

const fields = {
  totalEntries: document.querySelector("#totalEntries"),
  hitCount: document.querySelector("#hitCount"),
  missCount: document.querySelector("#missCount"),
  hitRate: document.querySelector("#hitRate"),
  cacheHit: document.querySelector("#cacheHit"),
  clusterId: document.querySelector("#clusterId"),
  matchScore: document.querySelector("#matchScore"),
  matchedQuery: document.querySelector("#matchedQuery"),
};

function percent(value) {
  return `${Math.round((value || 0) * 100)}%`;
}

function score(value) {
  return Number.isFinite(value) ? value.toFixed(3) : "-";
}

function escapeHtml(value) {
  return String(value || "").replace(/[&<>"']/g, (character) => {
    const entities = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    };
    return entities[character];
  });
}

function setStatus(state, text) {
  apiDot.className = `status-dot ${state}`;
  apiStatus.textContent = text;
}

function renderStats(stats) {
  fields.totalEntries.textContent = stats.total_entries ?? 0;
  fields.hitCount.textContent = stats.hit_count ?? 0;
  fields.missCount.textContent = stats.miss_count ?? 0;
  fields.hitRate.textContent = percent(stats.hit_rate);
}

async function loadStats() {
  try {
    const response = await fetch("/cache/stats");
    if (!response.ok) {
      throw new Error("Stats unavailable");
    }
    renderStats(await response.json());
    setStatus("ready", "Ready");
  } catch (error) {
    setStatus("error", "Needs Qdrant data");
  }
}

function renderResults(items) {
  if (!items.length) {
    results.innerHTML = '<p class="empty">No documents returned for this query.</p>';
    return;
  }

  results.innerHTML = items
    .map(
      (item) => `
        <article class="result">
          <header>
            <span class="category">${item.category || "uncategorized"}</span>
            <span class="score">${score(item.similarity_score)}</span>
          </header>
          <p>${escapeHtml(item.document_text)}</p>
        </article>
      `,
    )
    .join("");
}

function renderResponse(data) {
  fields.cacheHit.textContent = data.cache_hit ? "Hit" : "Miss";
  fields.clusterId.textContent = data.dominant_cluster ?? "-";
  fields.matchScore.textContent = score(data.similarity_score);
  fields.matchedQuery.textContent = data.matched_query
    ? `Matched cached query: "${data.matched_query}"`
    : "No cached query matched this request.";
  renderResults(data.results || []);
}

async function runQuery(query) {
  searchButton.disabled = true;
  searchButton.textContent = "Searching";
  setStatus("", "Searching...");

  try {
    const response = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });

    if (!response.ok) {
      const message = await response.json().catch(() => null);
      throw new Error(message?.detail || "Search failed");
    }

    renderResponse(await response.json());
    await loadStats();
  } catch (error) {
    setStatus("error", "Search unavailable");
    results.innerHTML = `<p class="empty">${error.message}</p>`;
  } finally {
    searchButton.disabled = false;
    searchButton.textContent = "Search";
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const query = input.value.trim();
  if (query) {
    runQuery(query);
  }
});

document.querySelectorAll("[data-query]").forEach((button) => {
  button.addEventListener("click", () => {
    input.value = button.dataset.query;
    runQuery(input.value);
  });
});

clearButton.addEventListener("click", async () => {
  clearButton.disabled = true;
  try {
    const response = await fetch("/cache", { method: "DELETE" });
    if (!response.ok) {
      throw new Error("Could not clear cache");
    }
    renderStats(await response.json());
    fields.cacheHit.textContent = "Pending";
    fields.clusterId.textContent = "-";
    fields.matchScore.textContent = "-";
    fields.matchedQuery.textContent = "";
  } finally {
    clearButton.disabled = false;
  }
});

loadStats();
