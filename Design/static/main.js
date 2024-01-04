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
      var $prediction = $("#prediction");

      // Xóa tất cả các lớp màu trước đó
      $prediction.removeClass("result-real result-fake result-unknown");

      if (prediction == 0) {
        // Nếu prediction = 0, thêm lớp màu xanh cho #prediction
        $prediction.text("Tin thật").addClass("result-real");
      } else if (prediction == 1) {
        // Nếu prediction = 1, thêm lớp màu đỏ cho #prediction
        $prediction.text("Tin giả").addClass("result-fake");
      }
    },
  });
});

// ... (Phần code xử lý chiều cao textarea ở đây)

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
