//Website Crime Counters
// const counter = document.querySelectorAll(".counter");
// setTimeout(() => {

//     counter.forEach((counter) => {
//         counter.innerHTML = 0;

//         const updateCounter = () => {
//             const targetCount = +counter.getAttribute('target');

//             const startingCount = Number(counter.innerHTML);

//             const inc = targetCount / 10;

//             if (startingCount < targetCount) {
//                 counter.innerHTML = Math.round(startingCount + inc);
//                 setTimeout(() => {
//                     updateCounter(counter);
//                 }, 500);
//             }
//         }

//         updateCounter(counter);
//     })
// }, 1000);
const counter = document.querySelectorAll(".counter");

setTimeout(() => {
    counter.forEach((counter) => {
        counter.innerHTML = 0;

        const updateCounter = () => {
            const targetCount = +counter.getAttribute('target');
            let startingCount = Number(counter.innerHTML);

            const inc = targetCount / 10;

            if (startingCount < targetCount) {
                startingCount = Math.min(startingCount + inc, targetCount);
                counter.innerHTML = Math.round(startingCount);

                if (startingCount < targetCount) {
                    setTimeout(() => {
                        updateCounter(counter);
                    }, 40);
                }
            }
        };

        updateCounter(counter);
    });
}, 1000);

// alert();

//vehicle hide code

let vehicle = document.getElementById("Vehicle");

function hide_vehicles(){

    // alert("hi")
    let selectedValue = vehicle.value;
    let hidevehicle = document.querySelectorAll(".hide_vehicle");

    if (selectedValue === "" || selectedValue === "None") {
        hidevehicle.forEach(function (element) {
            element.style.display = "none";
        });
    } else {
        hidevehicle.forEach(function (element) {
            element.style.display = "inline";
        });
    }
}

vehicle.addEventListener('change',hide_vehicles);

hide_vehicles()


// check pswd and cpswd
function checkPswd(event) {

    let pswd = document.querySelector("#password");
    let cpswd = document.querySelector("#cpassword");
    if (pswd.value === cpswd.value) {
        let form = document.querySelector("#register-user-form");
        form.submit();
        
    }
    else {

        let stat = document.querySelector("#status_register");
        stat.innerHTML = "password and confirm password not matched";
        stat.style.color = "red";
        event.preventDefault();
    }
}




//showpassword

function togglePassword() {
    const passwordInput = document.getElementById("exampleInputPassword1");
    const showPasswordCheckbox = document.getElementById("showPassword");
  
    // Toggle the input type between "password" and "text"
    passwordInput.type = showPasswordCheckbox.checked ? "text" : "password";
  }