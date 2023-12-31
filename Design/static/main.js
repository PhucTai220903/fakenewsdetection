$("form").on("submit", function (event) {
  event.preventDefault(); // Ngăn chặn hành vi mặc định của form
  var text = $("#text").val();
  if (!text) {
    alert("Vui lòng nhập nội dung bài báo!");
    return;
  }
  $.ajax({
    type: "POST",
    url: "/predict",
    data: { text: text },
    success: function (data) {
      var prediction = data.prediction;
      if (prediction == 0) {
        $("#prediction")
          .text("Tin thật")
          .addClass("result-real")
          .removeClass("result-fake result-unknown");
      } else if (prediction == 1) {
        $("#prediction")
          .text("Tin giả")
          .addClass("result-fake")
          .removeClass("result-real result-unknown");
      } else if (prediction == 2) {
        $("#prediction").text("The news cannot be classified");
      }
    },
  });
});

$("textarea").on("input", function () {
  // Đặt chiều cao tối đa bạn mong muốn (ví dụ: 200px)
  var maxHeight = 278;

  // Khi có sự kiện input (người dùng nhập liệu) xảy ra trên các thẻ <textarea>
  this.style.height = "auto"; // Đặt chiều cao của thẻ <textarea> thành "auto" để tránh giữ chiều cao cũ

  // Kiểm tra nếu chiều cao tính toán lớn hơn chiều cao tối đa
  if (this.scrollHeight > maxHeight) {
    this.style.height = maxHeight + "px"; // Đặt chiều cao của thẻ <textarea> thành giá trị tối đa nếu nó vượt quá
  } else {
    this.style.height = this.scrollHeight + "px"; // Ngược lại, sử dụng chiều cao tính toán
  }
  if (!this.value) {
    // Ẩn hoặc xóa kết quả dự đoán
    $("#prediction").text("");
  }
});
