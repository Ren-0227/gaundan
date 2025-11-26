import os

from pyarrow import deserialize, serialize

os.environ["KMP_WARNINGS"] = "FALSE" 

import json
import time
import warnings
from config import Config
from argparse import ArgumentParser
from functools import reduce
from multiprocessing import Process
from random import randint

import zmq
from ws4py.client.threadedclient import WebSocketClient

from utils.utils import *

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
parser = ArgumentParser()
parser.add_argument('--ip', type=str, default='localhost',
                    help='IP address of learner server')
parser.add_argument('--action_port', type=int, default=6000,
                    help='Learner server port to send training data')


RANK = {
    2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8,
    10:9, 11:10, 12:11, 13:12, 14:13
}


def _get_one_hot_array(num_left_cards, max_num_cards, flag):
    if flag == 0:     # 级数的情况
        one_hot = np.zeros(max_num_cards)
        one_hot[num_left_cards - 1] = 1
    else:
        one_hot = np.zeros(max_num_cards+1)    # 剩余的牌（0-1阵格式）
        one_hot[num_left_cards] = 1
    return one_hot


def _action_seq_list2array(action_seq_list):
    action_seq_array = np.zeros((len(action_seq_list), 54))
    for row, list_cards in enumerate(action_seq_list):
        action_seq_array[row, :] = card2array(list_cards)
    action_seq_array = action_seq_array.reshape(5, 216)
    return action_seq_array


def _process_action_seq(sequence, length=20):
    sequence = sequence[-length:].copy()
    if len(sequence) < length:
        empty_sequence = [[] for _ in range(length - len(sequence))]
        empty_sequence.extend(sequence)
        sequence = empty_sequence
    return sequence


class MyClient(WebSocketClient):
    def __init__(self, url, args):
        super().__init__(url)
        self.wins = 0
        self.games = 0
        self.args = args
        self.mypos = 0
        self.history_action = {0: [], 1: [], 2: [], 3:[]}
        self.action_seq = []
        self.action_order = [] # 记录出牌顺序(4个智能体是一样的)
        self.remaining = {0: 27, 1: 27, 2: 27, 3: 27}
        self.other_left_hands = [2 for _ in range(54)]
        self.flag = 0
        self.over = []
        self.rank = 1
        self.oppo_rank = 1

        self.cards = []

        # 初始化zmq
        self.context = zmq.Context()
        self.context.linger = 0
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f'tcp://localhost:{6000+args.client_index}')
        
    def opened(self):
        pass

    def closed(self, code, reason=None):
        print("Closed down", code, reason)

    def received_message(self, message):
        # 先序列化收到的消息，转为Python中的字典
        # message = json.loads(str(message))
        data = json.loads(str(message))
        operation = data.get("operation")
        phase = data.get("phase")
        msg_data = data.get("data", {})
        # print(message)
        
        # 牌局开始记录位置
        if operation == 'Deal':
            self.cards = msg_data["cards"]
            self.mypos = msg_data["position"]
        # 在动作序列中记录动作
        elif operation == 'PlayCard':
            play_position = msg_data["position"]
            cards = msg_data["cards"]
            just_play = play_position
            action = card2num(cards[2])
            if play_position != self.mypos:
                for ele in action:
                    self.other_left_hands[ele] -= 1
            if len(self.over) == 0:    # 如果没人出完牌
                self.action_order.append(just_play)
                self.action_seq.append(action)
                self.history_action[play_position].append(action)
            elif len(self.over) == 1:    # 只有一个出完牌的（如果队友也先赢了，就会直接结束）
                if len(action) > 0 and self.flag == 1: # 第一轮有人接下来了，则顺序没问题
                    self.flag = 2
                    if just_play == (self.over[0] + 3) % 4:     # 是头游的上家接下来的
                        self.action_order.append(just_play)       
                        self.action_seq.append(action)
                        self.history_action[play_position].append(action)
                        self.action_order.append(self.over[0])      # 添加第一个出完牌的玩家的信息
                        self.history_action[self.over[0]].append([-1])
                        self.action_seq.append([-1])
                    else:
                        self.action_order.append(just_play)        # 不是头游的上家接的
                        self.action_seq.append(action)
                        self.history_action[play_position].append(action)
                elif self.flag == 1 and (just_play + 1) % 4 == self.over[0]:      # 出完牌后全都没接的情况，由出完牌的对家出牌（如0、1、2、3、2）
                    self.flag = 2
                    self.action_order.append(just_play)        # 添加出完牌的上家
                    self.action_seq.append(action)
                    self.history_action[play_position].append(action)
                    self.action_order.append(self.over[0])      # 添加第一个出完牌的玩家的信息
                    self.history_action[self.over[0]].append([-1])
                    self.action_seq.append([-1])
                    self.action_order.append((just_play + 2) % 4)      # 添加被跳过出牌的玩家的信息
                    self.history_action[(just_play + 2) % 4].append([])
                    self.action_seq.append([])
                elif just_play == (self.over[0] + 3) % 4 and self.flag == 2:      # 当第一个出完牌的上家已经出过牌了(过完接风的第一轮后或有人接牌了)
                    self.action_order.append(just_play)        # 继续添加正常的信息
                    self.action_seq.append(action)
                    self.history_action[play_position].append(action)
                    self.action_order.append(self.over[0])      # 添加第一个出完牌的玩家的信息
                    self.history_action[self.over[0]].append([-1])
                    self.action_seq.append([-1])
                else:
                    self.action_order.append(just_play)        # 继续添加正常的信息
                    self.action_seq.append(action)
                    self.history_action[play_position].append(action)
            elif len(self.over) == 2:   # 可能包含两种情形（0、1和1、0出完情况不一样）
                if len(action) > 0 and self.flag <= 2:           # 有人接下来的情况
                    if (just_play+1) % 4 not in self.over:          # 下家牌没出完时，正常放过去
                        self.flag = 3        
                        self.action_order.append(just_play)        
                        self.action_seq.append(action)
                        self.history_action[play_position].append(action)    
                    else:
                        self.flag = 3
                        self.action_order.append(just_play)        # 是前二游玩家的上家接牌时
                        self.action_seq.append(action)
                        self.history_action[play_position].append(action)
                        self.action_order.append((just_play + 1) % 4)     # 先出完的肯定是紧挨着的上下家
                        self.history_action[(just_play + 1) % 4].append([-1])
                        self.action_seq.append([-1])
                        self.action_order.append((just_play + 2) % 4)     
                        self.history_action[(just_play + 2) % 4].append([-1])
                        self.action_seq.append([-1])     
                elif self.flag <= 2 and (just_play+1) % 4 in self.over:     # 接风时全都跳过的情况
                    self.flag = 3
                    self.action_order.append(just_play)        # 添加正常的信息
                    self.action_seq.append(action)
                    self.history_action[play_position].append(action)     
                    self.action_order.append((just_play + 1) % 4)     # 先出完的肯定是紧挨着的上下家
                    self.history_action[(just_play + 1) % 4].append([-1])
                    self.action_seq.append([-1])
                    self.action_order.append((just_play + 2) % 4)     
                    self.history_action[(just_play + 2) % 4].append([-1])
                    self.action_seq.append([-1])  
                    if just_play == (self.over[-1] + 2) % 4:  # 0、1情况 (1、0情况不用再加了)
                        self.action_order.append((just_play + 3) % 4)     
                        self.history_action[(just_play + 3) % 4].append([])
                        self.action_seq.append([])                             
                elif (just_play+1) % 4 in self.over and self.flag == 3: # 没出完牌的一定是上下家关系，当其中一个的下家出完时，就是两个出完的
                    self.action_order.append(just_play)        # 继续添加正常的信息
                    self.action_seq.append(action)
                    self.history_action[play_position].append(action)
                    self.action_order.append((just_play + 1) % 4)     # 先出完的肯定是紧挨着的上下家
                    self.history_action[(just_play + 1) % 4].append([-1])
                    self.action_seq.append([-1])
                    self.action_order.append((just_play + 2) % 4)     
                    self.history_action[(just_play + 2) % 4].append([-1])
                    self.action_seq.append([-1])
                else:
                    self.action_order.append(just_play)        # 继续添加正常的信息
                    self.action_seq.append(action)
                    self.history_action[play_position].append(action)

            self.remaining[just_play] -= len(action)
            if self.remaining[just_play] == 0:
                self.over.append(just_play)
        # 打牌
        elif operation == 'RequestAction':
            if self.flag == 0:       # 总共牌减去初始手牌
                init_hand = card2num(self.cards)
                for ele in init_hand:
                    self.other_left_hands[ele] -= 1
                self.flag = 1

            # 准备状态数据
            actions = msg_data.get("actions", [])

            if not actions or len(actions) == 1:
                # 没有可用动作，选择不出（pass）
                act_index = 0
            else :
                state = self.prepare(msg_data)
                # state = self.prepare(message)

                # print("send before")
                # 传输给决策模块
                self.socket.send(serialize(state).to_buffer())
                # print("send")
                # 收到决策
                act_index = deserialize(self.socket.recv())
                # print("recv after")

            response = {
                "operation": "Action",
                "actionIndex": act_index
            }
            # 作出决策
            self.send(json.dumps(response))

        # 小局结束
        elif operation == "GameResult":
            reward = self.get_reward(msg_data)
            if reward > 0:
                print("win: 我方获胜")
                self.wins += 1
            else:
                print("lose: 敌方获胜")
            self.games += 1
            print(self.wins, self.games, self.mypos)
            # 奖励数据传输至actor
            self.socket.send(serialize(reward).to_buffer())
            self.socket.recv()
            # 信息重置
            self.history_action = {0: [], 1: [], 2: [], 3:[]}
            self.action_seq = []
            self.other_left_hands = [2 for _ in range(54)]
            self.flag = 0
            self.action_order = []
            self.remaining = {0: 27, 1: 27, 2: 27, 3: 27}
            self.over = []
            self.cards = []

    def get_reward(self, message):
        team = [self.mypos, (self.mypos + 2) % 4]

        round_result = message["winList"]
        if round_result[self.mypos] == 1 or round_result[(self.mypos + 2) % 4] == 1:
            if round_result[self.mypos] == 2 or round_result[(self.mypos + 2) % 4] == 2:
                r = +3
            if round_result[self.mypos] == 3 or round_result[(self.mypos + 2) % 4] == 3:
                r = +2
            if round_result[self.mypos] == 4 or round_result[(self.mypos + 2) % 4] == 4:
                r = +1
        elif round_result[self.mypos] == 2 or round_result[(self.mypos + 2) % 4] == 2:
            if round_result[self.mypos] == 3 or round_result[(self.mypos + 2) % 4] == 3:
                r = -1
            if round_result[self.mypos] == 4 or round_result[(self.mypos + 2) % 4] == 4:
                r = -2
        else:
            r = -3

        return r

    def proc_universal(self, handCards, cur_rank):
        res = np.zeros(12, dtype=np.int8)

        if handCards[(cur_rank-1)*4] == 0:
            return res

        res[0] = 1
        rock_flag = 0
        for i in range(4):
            left, right = 0, 5
            temp = [handCards[i + j*4] if i+j*4 != (cur_rank-1)*4 else 0 for j in range(5)]
            while right <= 12:
                zero_num = temp.count(0)
                if zero_num <= 1:
                    rock_flag = 1
                    break
                else:
                    temp.append(handCards[i + right*4] if i+right*4 != (cur_rank-1)*4 else 0)
                    temp.pop(0)
                    left += 1
                    right += 1
            if rock_flag == 1:
                break
        res[1] = rock_flag

        num_count = [0] * 13
        for i in range(4):
            for j in range(13):
                if handCards[i + j*4] != 0 and i + j*4 != (cur_rank-1)*4:
                    num_count[j] += 1
        num_max = max(num_count)
        if num_max >= 6:
            res[2:8] = 1
        elif num_max == 5:
            res[3:8] = 1
        elif num_max == 4:
            res[4:8] = 1
        elif num_max == 3:
            res[5:8] = 1
        elif num_max == 2:
            res[6:8] = 1
        else:
            res[7] = 1
        temp = 0
        for i in range(13):
            if num_count[i] != 0:
                temp += 1
                if i >= 1:
                    if num_count[i] == 2 and num_count[i-1] >= 3 or num_count[i] >= 3 and num_count[i-1] == 2:
                        res[9] = 1
                    elif num_count[i] == 2 and num_count[i-1] == 2:
                        res[11] = 1
                if i >= 2:
                    if num_count[i-2] == 1 and num_count[i-1] >= 2 and num_count[i] >= 2 or \
                        num_count[i-2] >= 2 and num_count[i-1] == 1 and num_count[i] >= 2 or \
                        num_count[i-2] >= 2 and num_count[i-1] >= 2 and num_count[i] == 1:
                        res[10] = 1
            else:
                temp = 0
        if temp >= 4:
            res[8] = 1
        return res

    def prepare(self, message):
        num_legal_actions = len(message['actions'])
        legal_actions = [card2num(i['action'][2]) for i in message['actions']]
        my_handcards = card2array(card2num(message['cards']))   # 自己的手牌,54维
        # print('my_handcards', my_handcards)
        my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)

        universal_card_flag = self.proc_universal(my_handcards, RANK[message['rank']])     # 万能牌的标志位, 12维
        # print('universal_card_flag', universal_card_flag)
        universal_card_flag_batch = np.repeat(universal_card_flag[np.newaxis, :],
                                   num_legal_actions, axis=0)

        other_hands = []       # 其余所有玩家手上剩余的牌，54维
        for i in range(54): 
            if self.other_left_hands[i] == 1:
                other_hands.append(i)
            elif self.other_left_hands[i] == 2:
                other_hands.append(i)
                other_hands.append(i)
        # print(self.mypos, "other handcards: ", other_hands)
        other_handcards = card2array(other_hands)      
        # print('other_handcards', other_handcards)
        other_handcards_batch = np.repeat(other_handcards[np.newaxis, :],
                                      num_legal_actions, axis=0)

        last_action = []         # 最新的动作，54维
        if len(self.action_seq) > 0:
            last_action = card2array(self.action_seq[-1])
        else:
            last_action = card2array([-1])
        # print(last_action)
        last_action_batch = np.repeat(last_action[np.newaxis, :],
                                  num_legal_actions, axis=0)
        
        last_teammate_action = []               # 队友最后的动作， 54维
        if len(self.history_action[(self.mypos + 2) % 4]) > 0 and (self.mypos + 2) % 4 not in self.over:
            last_teammate_action = card2array(self.history_action[(self.mypos + 2) % 4][-1])
        else:
            last_teammate_action = card2array([-1])
        # print(last_teammate_action)
        last_teammate_action_batch = np.repeat(last_teammate_action[np.newaxis, :], num_legal_actions, axis=0)

        my_action_batch = np.zeros(my_handcards_batch.shape)     # 合法动作，54维
        for j, action in enumerate(legal_actions):
            my_action_batch[j, :] = card2array(action)

        down_num_cards_left = _get_one_hot_array(self.remaining[(self.mypos + 1) % 4], 27, 1)   # 下家剩余的牌数， 28维
        
        # print(down_num_cards_left)
        down_num_cards_left_batch = np.repeat(down_num_cards_left[np.newaxis, :], num_legal_actions, axis=0)

        teammate_num_cards_left = _get_one_hot_array(self.remaining[(self.mypos + 2) % 4], 27, 1)   # 对家剩余的牌数
        
        # print(teammate_num_cards_left)
        teammate_num_cards_left_batch = np.repeat(teammate_num_cards_left[np.newaxis, :], num_legal_actions, axis=0)

        up_num_cards_left = _get_one_hot_array(self.remaining[(self.mypos + 3) % 4], 27, 1)   # 上家剩余的牌数
        
        # print(up_num_cards_left)
        up_num_cards_left_batch = np.repeat(up_num_cards_left[np.newaxis, :], num_legal_actions, axis=0)

        if len(self.history_action[(self.mypos + 1) % 4]) > 0:
            down_played_cards = card2array(reduce(lambda x, y: x+y, self.history_action[(self.mypos + 1) % 4]))    # 下家打过的牌， 54维
        else:
            down_played_cards = card2array([])
        
        # print(down_played_cards)
        down_played_cards_batch = np.repeat(down_played_cards[np.newaxis, :], num_legal_actions, axis=0)

        if len(self.history_action[(self.mypos + 2) % 4]) > 0:
            teammate_played_cards = card2array(reduce(lambda x, y: x+y, self.history_action[(self.mypos + 2) % 4]))    # 对家打过的牌
        else:
            teammate_played_cards = card2array([])
        # print(teammate_played_cards)
        teammate_played_cards_batch = np.repeat(teammate_played_cards[np.newaxis, :], num_legal_actions, axis=0)

        if len(self.history_action[(self.mypos + 3) % 4]) > 0:
            up_played_cards = card2array(reduce(lambda x, y: x+y, self.history_action[(self.mypos + 3) % 4]))    # 上家打过的牌
        else:
            up_played_cards = card2array([])
        # print(up_played_cards)
        up_played_cards_batch = np.repeat(up_played_cards[np.newaxis, :], num_legal_actions, axis=0)
 
        self_rank = _get_one_hot_array(RANK[message['rank']], 13, 0)         # 己方当前的级牌，13维
        # print(self_rank)
        self_rank_batch = np.repeat(self_rank[np.newaxis, :], num_legal_actions, axis=0)

        oppo_rank = _get_one_hot_array(RANK[message['rank']], 13, 0)         # 敌方当前的级牌
        # print(oppo_rank)

        oppo_rank_batch = np.repeat(oppo_rank[np.newaxis, :], num_legal_actions, axis=0)

        cur_rank = _get_one_hot_array(RANK[message['rank']], 13, 0)         # 当前的级牌
        # print(cur_rank)

        cur_rank_batch = np.repeat(cur_rank[np.newaxis, :], num_legal_actions, axis=0)

        x_batch = np.hstack((my_handcards_batch,
                        universal_card_flag_batch,
                        other_handcards_batch,
                        last_action_batch,
                        last_teammate_action_batch,
                        down_played_cards_batch,
                        teammate_played_cards_batch,
                        up_played_cards_batch,
                        down_num_cards_left_batch,
                        teammate_num_cards_left_batch,
                        up_num_cards_left_batch,
                        self_rank_batch,
                        oppo_rank_batch,
                        cur_rank_batch,
                        my_action_batch))
        x_no_action = np.hstack((my_handcards,
                            universal_card_flag,
                            other_handcards,
                            last_action,
                            last_teammate_action,
                            down_played_cards,
                            teammate_played_cards,
                            up_played_cards,
                            down_num_cards_left,
                            teammate_num_cards_left,
                            up_num_cards_left,
                            self_rank,
                            oppo_rank,
                            cur_rank
                            ))
        obs = {
            'x_batch': x_batch.astype(np.float32),
            'legal_actions': legal_actions,
            'x_no_action': x_no_action.astype(np.float32),
        }
        return obs


def run_one_client(index, args):
    i0 = index // 4
    i1 = index % 4
    args.client_index = i1
    if i1 == 0:
        idx = "a1"
    elif i1 == 1:
        idx = "b1"
    elif i1 == 2:
        idx = "a2"
    elif i1 == 3:
        idx = "b2"
    # TODO:
    res = 23456 + i0
    client = MyClient(f'ws://localhost:{res}/{idx}', args)
    client.connect()
    client.run_forever()


def main():
    # 参数传递
    args, _ = parser.parse_known_args()

    def exit_wrapper(index, *x, **kw):
        """Exit all actors on KeyboardInterrupt (Ctrl-C)"""
        try:
            run_one_client(index, *x, **kw)
        except KeyboardInterrupt:
            if index == 0:
                for _i, _p in enumerate(clients):
                    if _i != index:
                        _p.terminate()

    clients = []
    if Config.is_train:
        for i in range(16 * 4):
            # print(f'start{i}')
            p = Process(target=exit_wrapper, args=(i, args))
            p.start()
            time.sleep(0.2)
            clients.append(p)
    else:
        for i in range(2):
            # print(f'start{i}')
            p = Process(target=exit_wrapper, args=(2 * i, args))
            p.start()
            time.sleep(0.2)
            clients.append(p)

    for client in clients:
        client.join()


if __name__ == '__main__':
    main()
