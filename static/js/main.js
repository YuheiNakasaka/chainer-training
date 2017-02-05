window.onload = function() {
  var can;
  var ct;
  var ox=0,oy=0,x=0,y=0;
  var mf=false;

  function mam_draw_init(){
    //初期設定
    can = document.getElementById("can");
    can.addEventListener("mousedown", onMouseDown, false);
    can.addEventListener("mousemove", onMouseMove, false);
    can.addEventListener("mouseup", onMouseUp, false);
    clearbtn = document.getElementById("clearbtn");
    clearbtn.addEventListener("click", clearCan, false);
    ct = can.getContext("2d");
    ct.strokeStyle="#000000";
    ct.lineWidth=5;
    ct.lineJoin="round";
    ct.lineCap="round";
    clearCan();
  }

  function onMouseDown(event){
    clearCan();
    ox=event.clientX-event.target.getBoundingClientRect().left;
    oy=event.clientY-event.target.getBoundingClientRect().top ;
    mf=true;
  }

  function onMouseMove(event){
    if(mf){
      x=event.clientX-event.target.getBoundingClientRect().left;
      y=event.clientY-event.target.getBoundingClientRect().top ;
      drawLine();
      ox=x;
      oy=y;
    }
  }

  function onMouseUp(event){
    mf=false;
    predict()
  }

  function drawLine(){
    ct.lineWidth = 23;
    ct.beginPath();
    ct.moveTo(ox,oy);
    ct.lineTo(x,y);
    ct.stroke();
  }

  function clearCan(){
    ct.fillStyle="rgb(255,255,255)";
    ct.fillRect(0,0,can.getBoundingClientRect().width,can.getBoundingClientRect().height);
    $('.result p').text('');
    $('.result').removeClass('highlight');
  }

  function predict() {
    var inputs = [];
    var can = document.getElementById("can");
    var img = new Image();
    img.onload = function() {
      var inputs = [];
      var small_canvas = document.createElement('canvas');
      small_canvas.width = 28;
      small_canvas.height = 28;
      var small = small_canvas.getContext('2d');
      small.drawImage(img, 0, 0, img.width, img.height, 0, 0, 28, 28);
      var data = small.getImageData(0, 0, 28, 28).data;
      for (var i = 0; i < 28; i++) {
          for (var j = 0; j < 28; j++) {
              var n = 4 * (i * 28 + j);
              inputs[i * 28 + j] = (data[n + 0] + data[n + 1] + data[n + 2]) / 3;
          }
      }
      if (Math.min(...inputs) === 255) {
          return;
      }
      $.ajax({
        type: "POST",
        url: "pred",
        contentType: 'application/json',
        data: JSON.stringify(inputs),
        success: function(data){
          console.log(data);
          var pred_all_mlp = JSON.parse(data["pred_all_mlp"]);
          for (var i = 0; i < pred_all_mlp.length; i++) {
            var pred = Math.round(pred_all_mlp[i]);
            $("#result.mlp .result."+i+" p").text(pred + "%");
            if (parseInt(data["pred_mlp"]) == i) {
              $("#result.mlp .result."+i).addClass('highlight');
            }
          }

          var max_cnn_idx = 0;
          var pred_all_cnn = JSON.parse(data["pred_all_cnn"]);
          for (var i = 0; i < pred_all_cnn.length; i++) {
            var pred = Math.round(pred_all_cnn[i]);
            $("#result.cnn .result."+i+" p").text(pred + "%");
            if (parseInt(data["pred_cnn"]) == i) {
              $("#result.cnn .result."+i).addClass('highlight');
            }
          }
        }
      });
    }
    img.src = can.toDataURL();
  }

  mam_draw_init();
}
