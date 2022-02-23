/*
    Copyright (c) 2022, David Turner <novalis@novalis.org>

     This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.

    This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
*/
'use strict';

let gameOver = false;
let firstGuess = true;
let guesses = [];
let guessed = new Set();
let guessCount = 0;
let model = null;
const now = Date.now();
const today = Math.floor(now / 86400000);
const initialDay = 19044;
const puzzleNumber = (today - initialDay) % secretWords.length;
const yesterdayPuzzleNumber = (today - initialDay + secretWords.length - 1) % secretWords.length;
const storage = window.localStorage;
let caps = 0;
let warnedCaps = 0;
let chrono_forward = 1;

function $(id) {
    if (id.charAt(0) !== '#') return false;
    return document.getElementById(id.substring(1));
}

function mag(a) {
    return Math.sqrt(a.reduce(function(sum, val) {
        return sum + val * val;
    }, 0));
}

function dot(f1, f2) {
    return f1.reduce(function(sum, a, idx) {
        return sum + a*f2[idx];
    }, 0);
}

function getCosSim(f1, f2) {
    return Math.abs(dot(f1,f2)/(mag(f1)*mag(f2)));
}


function plus(v1, v2) {
    const out = [];
    for (let i = 0; i < v1.length; i++) {
            out.push(v1[i] + v2[i]);
    }
    return out;
}

function minus(v1, v2) {
    const out = [];
    for (let i = 0; i < v1.length; i++) {
        out.push(v1[i] - v2[i]);
    }
    return out;
}


function scale (v, s) {
    const out = [];
    for (let i = 0; i < v.length; i++) {
        out.push(v[i] * s);
    }
    return out;
}


function project_along(v1, v2, t) {
    const v = minus(v2, v1);
    const num = dot(minus(t, v1), v);
    const denom = dot(v,v);
    return num/denom;
}

function share() {
    // We use the stored guesses here, because those are not updated again
    // once you win -- we don't want to include post-win guesses here.
    const text = solveStory(JSON.parse(storage.getItem("guesses")), puzzleNumber);
    const copied = ClipboardJS.copy(text);

    if (copied) {
        alert("Copied to clipboard");
    }
    else {
        alert("Failed to copy to clipboard");
    }
}

const words_selected = [];
const cache = {};
let secret = "";
let secretVec = null;
let similarityStory = null;

function select(word, secretVec) {
    /*
    let model;
    if (!(word in cache)) {
        // this can happen on a reload, since we do not store
        // the vectors in localstorage
        model = cache[word];
    } else {
        model = getModel(word);
        cache[word] = model;
    }
    words_selected.push([word, model.vec]);
    if (words_selected.length > 2) {
        words_selected.pop();
    }
    const proj = project_along(words_selected[0][1], words_selected[1][1],
                               target);
    console.log(proj);
*/
}

function guessRow(similarity, oldGuess, percentile, guessNumber, guess) {
    let percentileText = "(fr&iacute;o)";
    let progress = "";
    let cls = "";
    if (similarity >= similarityStory.rest * 100) {
        percentileText = '<span class="weirdWord">????<span class="tooltiptext">Unusual word found!  This word is not in the list of &quot;normal&quot; words that we use for the top-1000 list, but it is still similar! (Is it maybe capitalized?)</span></span>';
    }
    if (percentile) {
        if (percentile == 1000) {
            percentileText = "BUENA!";
        } else {
            cls = "close";
            percentileText = `<span style="text-align:right; width:5em; display:inline-block;">${percentile}/1000</span>&nbsp;`;
            progress = ` <span style="display:inline-block;width:10em;height:1ex; padding-bottom:1ex; background-color:#eeeeee;">
<span style="background-color:#008000; width:${percentile/10}%; display:inline-block">&nbsp;</span>
</span>`;
        }
    }
    let color;
    if (oldGuess === guess) {
        color = '#cc00cc';
    } else {
        color = '#000000';
    }
    return `<tr><td>${guessNumber}</td><td style="color:${color}" onclick="select('${oldGuess}', secretVec);">${oldGuess}</td><td>${similarity.toFixed(2)}</td><td class="${cls}">${percentileText}${progress}
</td></tr>`;

}

function updateLocalTime() {
    const now = new Date();
    now.setUTCHours(24, 0, 0, 0);

    $('#localtime').innerHTML = `o ${now.getHours()}:00 de tu zona horaria`;
}

function solveStory(guesses, puzzleNumber) {
    const guess_count = guesses.length;
    if (guess_count == 0) {
        return `Me rendí en el Semantle en español número ${puzzleNumber} sin siquiera intentar.`;
    }

    if (guess_count == 1) {
        return `Gané Semantle en español número ${puzzleNumber} en mi primer intento!`;
    }

    let describe = function(similarity, percentile) {
        let out = `con una similaridad de ${similarity.toFixed(2)}`;
        if (percentile) {
            out += ` (${percentile}/1000)`;
        }
        return out;
    }

    const guesses_chrono = guesses.slice();
    guesses_chrono.sort(function(a, b){return a[3]-b[3]});

    let [similarity, old_guess, percentile, guess_number] = guesses_chrono[0];
    let first_guess = `Mi primer intento fue ${describe(similarity, percentile)}.`;
    let first_guess_in_top = !!percentile;

    let first_hit = '';
    if (!first_guess_in_top) {
        for (let entry of guesses_chrono) {
            [similarity, old_guess, percentile, guess_number] = entry;
            if (percentile) {
                first_hit = `  Mi primera palabra en el top 1.000 fue al intento #${guess_number}.  `;
                break;
            }
        }
    }

    const penultimate_guess = guesses_chrono[guesses_chrono.length - 2];
    [similarity, old_guess, percentile, guess_number] = penultimate_guess;
    const penultimate_guess_msg = `Mi penúltimo intento fue ${describe(similarity, percentile)}.`;

    return `Resolví Semantle en español #${puzzleNumber} en ${guess_count} intentos. ${first_guess}${first_hit}${penultimate_guess_msg} http://semantle-es.cgk.cl/`;
}

let Semantle = (function() {
    async function getSimilarityStory(secret) {
        const url = "/similarity/" + secret;
        const response = await fetch(url);
        try {
            return await response.json();
        } catch (e) {
            return null;
        }
    }

    async function getModel(word) {
        if (cache.hasOwnProperty(word)) {
            return cache[word];
        }
        const url = "/model2/" + secret + "/" + word.replaceAll(" ", "_");
        const response = await fetch(url);
        try {
            return await response.json();
        } catch (e) {
            return null;
        }
    }

    async function getNearby(word) {
        const url = "/nearby/" + word ;
        const response = await fetch(url);
        try {
            return await response.json();
        } catch (e) {
            return null;
        }
    }

    async function init() {
        secret = secretWords[puzzleNumber].toLowerCase();
        const yesterday = secretWords[yesterdayPuzzleNumber].toLowerCase();

        $('#yesterday').innerHTML = `La palabra de ayer fue <b>"${yesterday}"</b>.`;
        $('#yesterday2').innerHTML = yesterday;

        try {
            const yesterdayNearby = await getNearby(yesterday);
            const secretBase64 = btoa(yesterday);
            $('#nearbyYesterday').innerHTML = `${yesterdayNearby.join(", ")}, en orden descendiente de cercan&iacute;a. <a href="nearby_1k/${secretBase64}">Más?</a>`;
        } catch (e) {
            $('#nearbyYesterday').innerHTML = `Ya viene!`;
        }
        updateLocalTime();

        try {
            similarityStory = await getSimilarityStory(secret);
            $('#similarity-story').innerHTML = `
Hoy, el puzzle número <b>${puzzleNumber}</b>, la palabra más cercana tiene una similaridad de 
<b>${(similarityStory.top * 100).toFixed(2)}</b>, la décima ${(similarityStory.top10 * 100).toFixed(2)} y la milésima ${(similarityStory.rest * 100).toFixed(2)}.
`;
        } catch {
            // we can live without this in the event that something is broken
        }

        const storagePuzzleNumber = storage.getItem("puzzleNumber");
        if (storagePuzzleNumber != puzzleNumber) {
            storage.clear();
            storage.setItem("puzzleNumber", puzzleNumber);
        }

        $('#give-up-btn').addEventListener('click', function(event) {
            if (!gameOver) {
                if (confirm("Seguro quieres rendirte?")) {
                    endGame(0);
                }
            }
        });

        $('#form').addEventListener('submit', async function(event) {
            event.preventDefault();
            if (secretVec === null) {
                secretVec = (await getModel(secret)).vec;
            }
            $('#guess').focus();
            $('#error').textContent = "";
            let guess = $('#guess').value.trim().replace("!", "").replace("*", "");
            if (!guess) {
                return false;
            }
            if ($("#lower").checked) {
                guess = guess.toLowerCase();
            }

            if (typeof unbritish !== undefined && unbritish.hasOwnProperty(guess)) {
                guess = unbritish[guess];
            }

            if (guess[0].toLowerCase() != guess[0]) {
                caps += 1;
            }
            if (caps >= 2 && (caps / guesses.length) > 0.4 && !warnedCaps) {
                warnedCaps = true;
                $("#lower").checked = confirm("You're entering a lot of words with initial capital letters.  This is probably not what you want to do, and it's probably caused by your phone keyboard ignoring the autocapitalize setting.  \"Nice\" is a city. \"nice\" is an adjective.  Do you want me to downcase your guesses for you?");
            }

            $('#guess').value = "";

            const guessData = await getModel(guess);
            if (!guessData) {
                $('#error').textContent = `No conozco la palabra ${guess}.`;
                return false;
            }

            let percentile = guessData.percentile;

            const guessVec = guessData.vec;

            cache[guess] = guessData;

            let similarity = getCosSim(guessVec, secretVec) * 100.0;
            if (!guessed.has(guess)) {
                guessCount += 1;
                guessed.add(guess);

                const newEntry = [similarity, guess, percentile, guessCount];
                guesses.push(newEntry);
            }
            guesses.sort(function(a, b){return b[0]-a[0]});
            saveGame(-1);

            chrono_forward = 1;

            updateGuesses(guess);

            firstGuess = false;
            if (guess.toLowerCase() === secret && !gameOver) {
                endGame(guesses.length);
            }
            return false;
        });

        const winState = storage.getItem("winState");
        if (winState != null) {
            guesses = JSON.parse(storage.getItem("guesses"));
            for (let guess of guesses) {
                guessed.add(guess[1]);
            }
            guessCount = guessed.size;
            updateGuesses("");
            if (winState != -1) {
                endGame(winState);
            }
        }
    }

    function updateGuesses(guess) {
        let inner = `<tr><th id="chronoOrder">#</th><th id="alphaOrder">Intento</th><th id="similarityOrder">Similaridad</th><th>Acercándote?</th></tr>`;
        /* This is dumb: first we find the most-recent word, and put
           it at the top.  Then we do the rest. */
        for (let entry of guesses) {
            let [similarity, oldGuess, percentile, guessNumber] = entry;
            if (oldGuess == guess) {
                inner += guessRow(similarity, oldGuess, percentile, guessNumber, guess);
            }
        }
        inner += "<tr><td colspan=4><hr></td></tr>";
        for (let entry of guesses) {
            let [similarity, oldGuess, percentile, guessNumber] = entry;
            if (oldGuess != guess) {
                inner += guessRow(similarity, oldGuess, percentile, guessNumber);
            }
        }
        $('#guesses').innerHTML = inner;
        $('#chronoOrder').addEventListener('click', event => {
            guesses.sort(function(a, b){return chrono_forward * (a[3]-b[3])});
            chrono_forward *= -1;
            updateGuesses(guess);
        });
        $('#alphaOrder').addEventListener('click', event => {
            guesses.sort(function(a, b){return a[1].localeCompare(b[1])});
            chrono_forward = 1;
            updateGuesses(guess);
        });
        $('#similarityOrder').addEventListener('click', event => {
            guesses.sort(function(a, b){return b[0]-a[0]});
            chrono_forward = 1;
            updateGuesses(guess);
        });
    }


    function saveGame(winState) {
        let oldState = storage.getItem("winState");
        if (oldState == -1 || oldState == null) {
            storage.setItem("winState", winState);
            storage.setItem("guesses", JSON.stringify(guesses));
        }
    }

    function endGame(guessCount) {
        $('#give-up-btn').style = "display:none;";
        $('#response').classList.add("gaveup");
        gameOver = true;
        const secretBase64 = btoa(secret);
        if (guessCount > 0) {
            $('#response').innerHTML = `<b>La encontraste en ${guessCount} intentos!  La palabra secreta es ${secret}</b>.  Puedes seguir intentando con otras palabras para ver su cercan&iacute;a. <a href="javascript:share();">Comparte</a> y juega otra vez mañana. Puedes ver las 1.000 palabras más cercanas <a href="nearby_1k/${secretBase64}">aquí</a>.`
        } else {
            $('#response').innerHTML = `<b>Te rendiste!  La palabra secreta es: ${secret}</b>.  Puedes seguir intentando con otras palabras para ver su cercan&iacute;a. <a href="javascript:share();">Comparte</a> y juega otra vez mañana. Puedes ver las 1.000 palabras más cercanas <a href="nearby_1k/${secretBase64}">aquí</a>.`;
        }
        saveGame(guessCount);
    }

    return {
        init: init
    };
})();
    
window.addEventListener('load', async () => { Semantle.init() });
