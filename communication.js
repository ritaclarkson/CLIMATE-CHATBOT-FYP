$(document).ready(function() {
    $("form").on("submit", function(event) {
        var userText = $("#text").val();
        var userHtml = '<div class="userText"><p>' + "You: <br> " + userText + "</p></div>";
        $("#text").val("");
        $("#chatbox").append(userHtml);
        document.getElementById("userInput").scrollIntoView({
            block: "start",
            behavior: "smooth",
        });
        $.ajax({
            data: {
                msg: userText,
            },
            type: "POST",
            url: "/get",
        }).done(function(data) {
            var botHtml = '<div class="botText"><p> Bot: <br>' + data + "</p></div>";
            $("#chatbox").append($.parseHTML(botHtml));
            document.getElementById("userInput").scrollIntoView({
                block: "start",
                behavior: "smooth",
            });
            var element = document.getElementById("chatbox");
            element.scrollTop = element.scrollHeight;
        });
        event.preventDefault();
    });

});