
// Simulate an asynchronous content loading process
 window.addEventListener('load', function() {
    // Simulate an API call or any asynchronous operation
    setTimeout(function() {
      document.querySelector('.loader-container').style.display = 'none'; // Hide loader container
    //   document.getElementById('content').style.display = 'block'; // Show content
    }, 500); // Adjust the timeout based on your needs
  });