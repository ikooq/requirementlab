const fields = [...document.querySelectorAll("textarea[data-finding]")];
const report = document.querySelector("#decisions").dataset.report;
const message = document.querySelector("#message");
let dirty = false;

function updateProgress() {
    const count = fields.filter(field => field.value.trim()).length;
    document.querySelector("#progress").textContent = `Записано решений: ${count} из ${fields.length}`;
}

for (const field of fields) {
    field.addEventListener("input", () => {
        dirty = true;
        updateProgress();
    });
}

document.querySelector("#save").addEventListener("click", () => {
    const answers = Object.fromEntries(fields.map(field => [field.dataset.finding, field.value]));
    const file = new Blob([JSON.stringify({report, answers}, null, 2)], {type: "application/json"});
    const url = URL.createObjectURL(file);
    const link = document.createElement("a");
    link.href = url;
    link.download = `requirementlab-answers-${report.slice(0, 12)}.json`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    dirty = false;
});

document.querySelector("#restore").addEventListener("change", async event => {
    const input = event.target;
    const file = input.files[0];
    if (!file) return;
    try {
        const saved = JSON.parse(await file.text());
        if (saved.report !== report || !saved.answers || typeof saved.answers !== "object" ||
            Object.keys(saved.answers).length !== fields.length ||
            !fields.every(field => typeof saved.answers[field.dataset.finding] === "string")) {
            throw new Error("Файл повреждён или относится к другому отчёту.");
        }
        if (dirty && !window.confirm("Заменить введённые ответы данными из файла?")) return;
        for (const field of fields) field.value = saved.answers[field.dataset.finding];
        dirty = false;
        updateProgress();
        message.textContent = "Ответы загружены.";
    } catch (error) {
        message.textContent = "Не удалось загрузить ответы. " + error.message;
    } finally {
        input.value = "";
    }
});

window.addEventListener("beforeunload", event => {
    if (dirty) {
        event.preventDefault();
        event.returnValue = "";
    }
});
updateProgress();
