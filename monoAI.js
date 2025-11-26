let prompt = document.getElementById('msg')

function chat() {
  document.writeln(prompt)
    if (input === 'hi') {
      console.log('MonoAI: hello');
    } else if (input === 'mono') {
        console.log('MonoAI: yo aswin');
    } else {
      console.log('MonoAI: mono');
    }
    
    chat();
  };

console.log('Chat started! (Press Ctrl+C to exit)');
chat();