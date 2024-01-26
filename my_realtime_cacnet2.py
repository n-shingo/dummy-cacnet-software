# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 14:39:42 2024

@author: shingo



"""
import sys
import math
import types
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from config_classification import cfg
from my_CACNet import MyCACNet
from PIL import Image
from cac_software_uitl import DrawTool
from KUPCP_dataset import IMAGE_NET_MEAN, IMAGE_NET_STD
# import torchvision.transforms as transforms
# from config_classification import cfg
# from KUPCP_dataset import IMAGE_NET_MEAN, IMAGE_NET_STD


#カメラの取り込み(HDMI)指定
CAM_ID= 0
#HDMI_W, HDMI_H = 1920, 1080  # FHD 1920 x 1080
HDMI_W, HDMI_H = 1280, 720  # HD  1280 x 720

# 表示用画像サイズ
SHOW_H = HDMI_H
SHOW_W = SHOW_H * 3 // 2
HDMI_PAD = (HDMI_W-SHOW_W) // 2

# 処理用画像サイズ
PRCS_H = 300
PRCS_W = PRCS_H * 3 // 2

# SIFT & FLANN (特徴点マッチング関連)
SIFT = cv2.SIFT_create()
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
FLANN = cv2.FlannBasedMatcher( index_params, search_params)

# Adjust 閾値
TH_DIRECT = 0.02
TH_ANGLE = 1.0
TH_ZOOM = 0.03


# GPU or CPU
GPU = True
if GPU:
    DEVICE = torch.device('cuda:0')
else:
    DEVICE = torch.device('cpu')


# CACNet
TRANS = transforms.Compose([
    transforms.Resize((cfg.image_size[0], cfg.image_size[1])),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
])

# cv2.waitKey 用番号
ENTER = 13
ESC = 27

def main():
    
    # 表示用フラグ
    flags = types.SimpleNamespace()
    flags.idle_sift = False # siftキーポイント表示
    flags.idle_info = False # 画像情報表示
    flags.indicators = True # インジケータの表示
    
    # カメラキャプチャ作成
    cap = cv2.VideoCapture(CAM_ID, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print('ビデオ(ID:', CAM_ID, ')が開けません')
        sys.exit()
    
    # カメラ設定
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HDMI_H)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, HDMI_W)
    print( "取込画像サイズ: ( {0}, {1} )".format(HDMI_W, HDMI_H))
    print( "表示画像サイズ: ( {0}, {1} )".format(SHOW_W, SHOW_H))
    print( "処理画像サイズ: ( {0}, {1} )".format(PRCS_W, PRCS_H))
    
    
    # CACNet 準備
    weight_file = "D:/Eriko/CACNet-Pytorch/experiments/231222_loss_bonus_4/checkpoints/best-FCDB_disp.pth"
    model = MyCACNet(loadweights=False, bonus_pos=4)
    # if GPU:
    #     model.load_state_dict(torch.load(weight_file, map_location="cuda:0"))
    # else:
    #     model.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
    model.load_state_dict(torch.load(weight_file, map_location=DEVICE))
    model = model.to(DEVICE).eval()
    
    # mode
    mode = 'idling'  # idling / adjust
    
    # キャプチャ時の取得変数
    capt_img = None  # キャプチャした時の画像
    capt_kpts = None # キャプチャ画像の SIFT キーポイント
    capt_desc = None # キャプチャ画像の SIFT 特徴量
    crop_rect = None  # キャプチャした画像の切り出し領域
    
    
    # 画像取込
    while True:
    
        # 画像取得
        ret, frame = cap.read()
        
        # 表示用と処理用画像を作成
        show_img, prcs_img = make_show_and_prc_images(frame)
        
        if mode == 'idling':
            show_img = draw_idling_display( show_img, flags )
            
        elif mode == 'adjust':
        
            # sift特徴量取得
            kp, des = SIFT.detectAndCompute( prcs_img, None )
            
            # キャプチャ画像から現在の画像に変換するホモグラフィ行列を求める
            H = homography_matrix( capt_kpts, capt_desc, kp, des )

            # ディスプレイ描画
            show_img = draw_adjust_display( show_img, crop_rect, H, flags )
    

        # 画像表示        
        cv2.imshow('Image', show_img)
    
        # キー入力処理
        key =cv2.waitKey(1)
        
        # Quit
        if key == ESC or key == ord('q'):
            break
        
        # idling モード時
        if mode == 'idling':

            if key == ENTER:
                # キャプチャ & CACNet実行
                capt_img = prcs_img.copy()
                crop_rect = get_cropping_rect( model, prcs_img )  # 切取領域
    
                # SIFT
                capt_kpts, capt_desc = SIFT.detectAndCompute(capt_img, None) # キャプチャ画像のSIFT特徴量
                print( 'Feature key point size: ', capt_desc.shape )  # (キーポイント数 x 特徴量次元数)
                
                # adjust モードへ
                mode = 'adjust'
                
            if key == ord('s'):
                flags.idle_sift = not flags.idle_sift
        
        # adjust モード時
        elif mode == 'adjust':
            
            if key == ENTER:
                # idling モードへ
                mode = 'idling'
                
            if key == ord('i'):
                flags.indicators = not flags.indicators
        
    
    # 終了作業
    cap.release()
    cv2.destroyAllWindows()


def get_cropping_rect( model, img ):

    # CACNet
    # aspect比
    img_h, img_w= img.shape[:2]
    aspect = img_h / img_w
    aspect = aspect*img_w/img_h*cfg.image_size[0]/cfg.image_size[1]
    aspect = torch.Tensor([[aspect]]).to(DEVICE)
    
    # CACNet用入力データに変換
    input_image = Image.fromarray(img)
    input_image = TRANS( input_image )
    input_image = input_image.unsqueeze(dim=0).to(DEVICE)
    
    # MyCACNet に入力
    logits,kcm,box = model(input_image , aspect )
    
    # 切り出し領域の計算
    box = box.detach().cpu()
    x1, y1, x2, y2 = box[0].tolist()
    x1 *= img_w / cfg.image_size[1]
    x2 *= img_w / cfg.image_size[1]
    y1 *= img_h / cfg.image_size[0]
    y2 *= img_h / cfg.image_size[0]

    # 終了
    crop_rect = [(x1,y1), (x1,y2), (x2,y2), (x2,y1)]
    return crop_rect


def homography_matrix( kp1, des1, kp2, des2 ):
    
    '''
    ２つのSIFT特徴量の計算結果から 1 から 2 へのホモグラフィ行列を求める
    失敗した場合は None を返す

    SIFT プログラムの参考
    https://qiita.com/suuungwoo/items/9598cbac5adf5d5f858e
    http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_feature2d/py_matcher/py_matcher.html

    '''
    
    if kp1 is None or des1 is None or kp2 is None or des2 is None:
        return None
    
    if len(kp1) < 5 or len(kp2) <5 :
        return None
    
    # k近傍法でマッチング(k=2)
    matches = FLANN.knnMatch(des1, des2, k=2)
    
    # 良いマッチングを抽出
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append([m])
    
    # 十分な数がなければ失敗・終了
    if len(good_matches) < 12:
        return None

    # 対応するキーポイントのリスト作成
    kpts1 = np.float32(
        [kp1[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    kpts2 = np.float32(
        [kp2[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # ホモグラフィを計算
    H, status = cv2.findHomography(kpts1, kpts2, cv2.RANSAC, 5.0)
    cnt = np.count_nonzero(status)
   
    # マッチング情報
    match_info = types.SimpleNamespace()
    match_info.H = H  # ホモグラフィ行列
    match_info.kpts_cnts = ( len(kp1), len(kp2) )  # SIFTのキーポイント数
    match_info.good_cnts = len(good_matches) # 良好なマッチング数
    match_info.valid_cnts = cnt # RANZAC による有効なマッチング数

    
    # 終了
    return H


def draw_idling_display( img, flags ):
    
    '''
    Idling モード時の描画
    '''
    
    # 横幅1920pixを基準としたscale
    img_h, img_w= img.shape[:2]
    
    # SIFTキーポイント描画
    if flags.idle_sift:
        draw_sift(img)

    # PIL で描画    
    pil_img = DrawTool.to_pil_image(img)
    
    # ヘッダ（背景, 警告マーク， テキスト）描画
    msg = '撮影したい方向にカメラを向けて Enter を押してください'
    draw_header(pil_img, msg, color=(255,50,50))
    
    # footer
    draw_footer( pil_img, mode='idling' )

    # numpy 配列に戻す
    cv2_img = DrawTool.to_cv2_image(pil_img)
    
    return cv2_img


def draw_adjust_display( img, crop_rect, h_matrix, flags ):
    
    '''
    Adjust モード時の描画
    '''

    # 画像サイズ
    img_h, img_w = img.shape[:2]
    

    # 諸々の値の計算
    outer_rect = None   # 外枠
    inner_rect = None  # 内枠
    matched = False # 全体のマッチング
    direct_matched = False # 方向のマッチング
    angle_matched = False # 水平角度のマッチング
    zoom_matched = False # ズームのマッチング
    direct_dif = None
    angle_dif = None
    zoom_dif = None
    
    if not h_matrix is None:
        
        # 処理用と表示用の比率
        scale = img_h/PRCS_H
        
        # ホモグラフィー変換する点群作成
        outer_rect = np.array( crop_rect )
        center= (outer_rect[2]+outer_rect[0]) / 2
        inner_rect = (outer_rect + center) / 2
        mid_pnts = [ (inner_rect[i]+inner_rect[(i+1)%4])/2 for i in range(4) ]
        pnts = np.concatenate( [outer_rect, inner_rect, mid_pnts, center.reshape(1,2) ], axis=0 )
        
        # 変換
        pnts = pnts.reshape( 1,-1, 2 )
        pnts = cv2.perspectiveTransform( pnts, h_matrix )
        pnts = pnts.squeeze(0)
        pnts = pnts * scale
        
        # 取り出し
        outer_rect = pnts[0:4,:]
        inner_rect = pnts[4:8,:]
        mid_pnts = pnts[8:12]
        center = pnts[12]
        
        # ズレ計算
        direct_dif = cal_difference_direction( center, (img_w, img_h) )  # 位置(dx, dy)
        angle_dif = cal_difference_angle_deg( mid_pnts ) # 水平角度
        zoom_dif  = cal_difference_zoom( mid_pnts, img_w )  # 倍率(0でズレなし)
        
        # マッチング
        direct_matched = match_direction(direct_dif) 
        angle_matched = match_angle(angle_dif) 
        zoom_matched =  match_zoom(zoom_dif)
        matched = direct_matched and angle_matched and zoom_matched
        
        # 描画用に Tuple へ変換
        outer_rect = [tuple(e) for e in outer_rect.astype('int')]
        inner_rect = [tuple(e) for e in inner_rect.astype('int')]

    ### ここから描画 ####        
        
    # 横幅1920pixを基準としたscale
    s = img_w/1920

    # マッチしていないときは画面を暗めにする
    if not matched:
        img = (img * 0.5).astype('uint8')
    
    # 中心矩形描画
    pt1 = (img_w // 4, img_h // 4)
    pt2 = (img_w // 4, img_h * 3 // 4)
    pt3 = (img_w * 3 // 4, img_h * 3//4 )
    pt4 = (img_w * 3 // 4, img_h // 4 )
    ref_rect = ( pt1, pt2, pt3, pt4 )
    for i in range(4) :
        cv2.line( img, ref_rect[i-1], ref_rect[i], (0,180,0), thickness=6, lineType=cv2.LINE_AA)

    # 外と内側枠線描画        
    if outer_rect is not None:
        color = (0,255,0) if matched else (0,0,255)
        for i in range(4):
            cv2.line( img, outer_rect[i-1], outer_rect[i], (200,200,200), thickness=2, lineType=cv2.LINE_AA)
            DrawTool.dashed_line( img, inner_rect[i-1], inner_rect[i], gap=8, linewidth=2, color=color)
         
    # インジケータ描画
    if flags.indicators:
        ind_y = img_h - s*220
        ind_w = int(s*380)
        ind_x = img_w/2 - ind_w/2
        DrawTool.direct_gadget(img, (ind_x-ind_w, ind_y), direct_dif, ind_w, TH_DIRECT )
        DrawTool.angle_gadget(img, (ind_x, ind_y), angle_dif, ind_w, angle_matched)
        DrawTool.zoom_gadget(img, (ind_x+ind_w, ind_y), zoom_dif, ind_w, zoom_matched )
            
    # ここから PIL Image で情報描画
    
    # PIL で描画    
    #pil_img = Image.fromarray(img[:, :, [2, 1, 0]]).convert('RGBA') # BGR->RGBA
    pil_img = DrawTool.to_pil_image(img)
    
    # ヘッダ（背景, 警告マーク テキスト描画）
    msg = '緑の枠と赤い点線枠を重ねてください'
    draw_header(pil_img, msg, color=(255,255,255))

    # シャッター指示の表示
    if matched:
        msg = 'シャッターを押してください！'
        w_2, h_2 = img_w/2, img_h/2
        DrawTool.fillrect( pil_img, (w_2 - s*600, h_2-s*80, w_2 + s*600, h_2+s*80), fill=(0,0,0, 200) )
        DrawTool.information_icon(pil_img, (w_2-s*580, h_2-s*40), s*70)
        DrawTool.text( pil_img, (w_2+s*20, h_2), msg, size=s*70, color=(255,255,255), anchor='mm' )

    # footer
    draw_footer( pil_img, mode='adjust' )

    # cv2 画像に戻す
    cv2_img = DrawTool.to_cv2_image(pil_img)

    # 終了
    return cv2_img


def draw_sift( img ):
    sift = cv2.SIFT_create(contrastThreshold=0.18)  # 数を減らすために閾値大きめ
    kps, des = sift.detectAndCompute(img, None)
    img[:] = cv2.drawKeypoints(img, kps, None, flags=4, color=(255,0,0))[:] # inplace で描画
    
def draw_header( pil_img, msg, color ):
    img_w = pil_img.width
    s = img_w / 1920
    DrawTool.fillrect(pil_img, (0, 0, img_w, int(70*s)), fill=(0,0,0,200))
    DrawTool.warning_icon( pil_img, (s*20,s*10), s*48 )
    DrawTool.text( pil_img, (s*90, s*0.01), msg , size=s*50, color=color )


def draw_footer( img, mode ):

    # 横幅1920pixを基準としたscale
    w, h= img.size
    s = w/1920

    # 背景
    DrawTool.fillrect( img, (0, h-s*50, w, h), fill=(0,0,0, 200) )
    
    # テキスト
    if mode == 'idling':
        text = "[Enter] Adjust モード     [R] 回転      [A] 画像情報     [S] SIFTキーポイント     [ESC] 終了"
    elif mode == 'adjust':
        text = "[Enter] Idling モード     [A] 調整アドバイス     [I] インジケータ     [M] マッチング情報     [C] CACNet-fix 情報    [ESC] 終了"
    else:
        text = 'mode "' + mode + '" is not defined.'
    DrawTool.text( img, (w/2, h), text, size=s*30, color=(255,255,255), anchor='md')
    
    
def cal_difference_direction( center, img_size ) :
    
    '''
    中心点のズレを計算
    ズレ1は画像中心から短い端までの距離
    '''
    
    w, h = img_size
    length = min(w, h)
    
    dx = center[0]-w/2
    dy = center[1]-h/2
    
    return( dx/length, dy/length )
        
def cal_difference_angle_deg( mid_pnts ) :

    '''
    水平角度のズレ角度(deg)を計算
    '''
    lft = mid_pnts[0]
    rgt = mid_pnts[2]
    dx = rgt[0]-lft[0]
    dy = rgt[1]-lft[1]
    angle = math.atan2(dy,dx)
    angle *= 180/math.pi
    return angle

def cal_difference_zoom( mid_pnts, img_width ) :

    '''
    ズームのズレを計算
    0だとピッタリ
    '''
    lft = mid_pnts[0]
    rgt = mid_pnts[2]
    dx = rgt[0]-lft[0]
    dy = rgt[1]-lft[1]
    d = (dx*dx+dy*dy)**0.5
    
    half_w = img_width / 2
    rate = d / half_w
    return math.log(rate)


def make_show_and_prc_images( hdmi_img ):
    '''
    HDMIの生画像から、表示用と初利用の画像を生成する
    '''
    shw_img = hdmi_img[:,HDMI_PAD:SHOW_W+HDMI_PAD]
    prc_img = cv2.resize(shw_img, (PRCS_W, PRCS_H))
    return shw_img, prc_img


def match_direction( direct_dif ):
    dx, dy = direct_dif
    d = (dx*dx+dy*dy)**0.5
    res = d < TH_DIRECT
    return res

def match_angle( angle_dif ):
    res = abs(angle_dif) < TH_ANGLE
    return res

def match_zoom( zoom_dif ):
    res = abs(zoom_dif) < TH_ZOOM
    return res


if __name__ == '__main__':
    main()
    
    