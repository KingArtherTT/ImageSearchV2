var video_path = {}
video_path['000006'] = '<iframe src="http://open.iqiyi.com/developer/player_js/coopPlayerIndex.html?vid=98335dd0228c4293d08f47beb85f644a&tvId=16035785900&accessToken=2.f22860a2479ad60d8da7697274de9346&appKey=3955c3425820435e86d0f4cdfe56f5e7&appId=1368&height=100%&width=100%" frameborder="0" allowfullscreen="true" width="100%" height="100%"></iframe>'
video_path['000044'] = '<iframe src="http://open.iqiyi.com/developer/player_js/coopPlayerIndex.html?vid=b579b42749d8e0a46afd58d1d51f1454&tvId=16040236000&accessToken=2.f22860a2479ad60d8da7697274de9346&appKey=3955c3425820435e86d0f4cdfe56f5e7&appId=1368&height=100%&width=100%" frameborder="0" allowfullscreen="true" width="100%" height="100%"></iframe>'
video_path['000054'] = '<iframe src="http://open.iqiyi.com/developer/player_js/coopPlayerIndex.html?vid=e6ef281e71dc8879fce623b165dceb8a&tvId=16040243600&accessToken=2.f22860a2479ad60d8da7697274de9346&appKey=3955c3425820435e86d0f4cdfe56f5e7&appId=1368&height=100%&width=100%" frameborder="0" allowfullscreen="true" width="100%" height="100%"></iframe>'
video_path['000077'] = '<iframe src="http://open.iqiyi.com/developer/player_js/coopPlayerIndex.html?vid=50d3320a463878465debb859f2994a4f&tvId=16040251700&accessToken=2.f22860a2479ad60d8da7697274de9346&appKey=3955c3425820435e86d0f4cdfe56f5e7&appId=1368&height=100%&width=100%" frameborder="0" allowfullscreen="true" width="100%" height="100%"></iframe>'
video_path['000078'] = '<iframe src="http://open.iqiyi.com/developer/player_js/coopPlayerIndex.html?vid=f5052c6640f6a8e1de5bb018a03f8298&tvId=16040260700&accessToken=2.f22860a2479ad60d8da7697274de9346&appKey=3955c3425820435e86d0f4cdfe56f5e7&appId=1368&height=100%&width=100%" frameborder="0" allowfullscreen="true" width="100%" height="100%"></iframe>'


String.prototype.format = function (args) {
    if (arguments.length > 0) {
        var result = this;
        if (arguments.length == 1 && typeof (args) == "object") {
            for (var key in args) {
                var reg = new RegExp("({" + key + "})", "g");
                result = result.replace(reg, args[key]);
            }
        } else {
            for (var i = 0; i < arguments.length; i++) {
                if (arguments[i] == undefined) {
                    return "";
                } else {
                    var reg = new RegExp("({[" + i + "]})", "g");
                    result = result.replace(reg, arguments[i]);
                }
            }
        }
        return result;
    } else {
        return this;
    }
}


function get_result(video_id) {
    $('#single_result_list').css('display', 'block')
    $('#final_result').css('display', 'block')
    // 发起request 请求，返回四张图片的文件名称
    var data = {}
    data['video_id'] = video_id
    $.ajax({
        type: 'POST',
        url: '/action/get_result/',
        data: JSON.stringify(data),
        dataType: 'json',
        contentType: 'application/json; charset=UTF-8',
        success: function (result) {
            console.log(result)
            if (parseInt(result['code']) == 1) {
                //替换照片
                var appendHTML = ''
                for (i = 0; i < result['top_list'].length; i++) {
                    appendHTML += '<div class="row"><a href="javascript:void(0);" class="thumbnail">'
                    appendHTML += '<img src="/static/data/dealed_image/{0}" alt="{1}">'.format(result['top_list'][i]['img_path'], result['top_list'][i]['img_path'])
                    appendHTML += '<div class="caption">与第{0}帧的距离：{1}'.format(result['top_list'][i]['frame_index'], result['top_list'][i]['distance'])
                    appendHTML += '</div></a></div>'
                }
                $('#single_result_list').html('<div class="row modal-header"><h3>单帧匹配结果展示(每隔40帧匹配一次)</h3></div>')
                $('#single_result_list').html($('#single_result_list').html() + appendHTML)
                $('#final_result img').attr('src', "/static/data/dealed_image/{0}".format(result['final_result']['img_path']))
                $('#final_result img').attr('alt', result['final_result']['img_path'])
                $('#final_result .caption').html('与第{0}帧的距离：{1}'.format(result['final_result']['frame_index'], result['final_result']['distance']))
            } else {
                console.log(result['message'])
                alert(result['message'])
            }
        }
    })
}

function refresh_search() {
    var video_id = $('#video').attr('src')
    var i = 0
    for (var k in video_path) {
        if (k == video_id) {
            break
        }
        ++i
    }
    i = (i + 1) % Object.keys(video_path).length
    video_id = Object.keys(video_path)[i]

    $('#video').html(video_path[video_id])
    $('#video').attr('src', video_id)
    get_result(video_id)

}

$(document).ready(function () {
    $('#refresh').on('click', refresh_search)
    get_result($('#video').attr('src'))
})


