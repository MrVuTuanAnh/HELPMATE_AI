function submitQuestion() {
    var question = document.getElementById("question").value;
    fetch('/get-answer', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: question }),
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("answer").innerText = data.answer;
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}
