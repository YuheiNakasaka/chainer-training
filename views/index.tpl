<html>
  <head>
    <meta charset="utf-8">
    <title>MNIST Web</title>
    <script src="https://code.jquery.com/jquery-3.1.1.min.js" integrity="sha256-hVVnYaiADRTO2PzUGmuLJr8BLUSjGIZsDYGmIJLv2b8=" crossorigin="anonymous"></script>
    <style>
    #wrap {
      float: left;
    }
    #wrap canvas {
      border: 2px solid #000;
    }
    #preview {
      width: 100px;
      display: inline-block;
    }
    #preview img {
      width: 28px;
      height: 28px;
    }
    #result {
      margin: 0 0 0 20px;
      float: left;
    }
    #result h2 {
      margin: 0;
      display: inline-block;
      font-size: 16px;
    }
    #result p {
      margin: 0;
      display: inline-block;
    }
    #buttons {
      clear: both;
      margin: 0;
    }
    </style>
  </head>
  <body>
    <h1>MNIST Web with Chainer</h1>
    <div id="wrap">
      <canvas id="can" width="400" height="400"></canvas>
    </div>
    <div id="result">
      <div class="result 0">
        <h2>0</h2>
        <p></p>
      </div>
      <div class="result 1">
        <h2>1</h2>
        <p></p>
      </div>
      <div class="result 2">
        <h2>2</h2>
        <p></p>
      </div>
      <div class="result 3">
        <h2>3</h2>
        <p></p>
      </div>
      <div class="result 4">
        <h2>4</h2>
        <p></p>
      </div>
      <div class="result 5">
        <h2>5</h2>
        <p></p>
      </div>
      <div class="result 6">
        <h2>6</h2>
        <p></p>
      </div>
      <div class="result 7">
        <h2>7</h2>
        <p></p>
      </div>
      <div class="result 8">
        <h2>8</h2>
        <p></p>
      </div>
      <div class="result 9">
        <h2>9</h2>
        <p></p>
      </div>
    </div>
    <div id="buttons">
      <button id="clearbtn">全部消す</button>
    </div>
    <script type="text/javascript" src="static/js/main.js" charset="utf-8"></script>
  </body>
</html>
