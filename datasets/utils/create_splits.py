import os
import re
import datetime
import numpy as np
import pandas as pd

ek_train_classes = [
    'take', 'put-down', 'open', 'turn-off',
    'dry', 'hand', 'tie', 'remove', 'cut', 'pull-down', 'shake',
    'drink', 'move', 'lift', 'stir', 'adjust', 'crush', 'taste',
    'check', 'drain', 'sprinkle', 'empty', 'knead', 'spread-in',
    'scoop', 'add', 'push', 'set-off', 'wear', 'fill', 'turn-down',
    'measure', 'scrape', 'read', 'peel', 'smell', 'plug-in', 'flip',
    'turn', 'enter', 'unscrew', 'screw-in', 'tap-on', 'break',
    'fry', 'brush', 'scrub', 'spill', 'separate', 'immerse', 'rub-on', 
    'lower', 'stretch', 'slide', 'use', 'form-into', 'oil',
    'sharpen', 'touch', 'let',
]

ek_test_classes = [
    'wash', 'squeeze', 'turn-on', 'throw-in', 'close', 'put-into', 
    'fold', 'unfold', 'pour', 'tear', 'look-for', 'hold', 'roll', 
    'arrange', 'spray', 'wait', 'collect', 'turn-up', 'grate', 'wet'
]

# def add_noaction(df, video_info, min_size=0):
#     nadf = pd.concat([df, df.iloc[-1:]]).sort_values('start_frame')
#     nadf['src_narration_id'] = nadf.narration_id.copy()
#     nadf.narration_id.iloc[-1] += '_final'
#     nadf['narration_id'] = nadf.narration_id.apply(lambda x: f'{x}_noaction')
#     nadf['verb'] = ''
#     nadf['noun'] = ''
#     nadf['verb_class'] = -1
#     nadf['noun_class'] = -1
#     nadf['all_nouns'] = nadf.all_nouns.apply(lambda x: [])
#     nadf['all_noun_classes'] = nadf.all_nouns.apply(lambda x: [])

#     duration = video_info.duration[df.name]
#     fps = video_info.fps[df.name]
#     nadf['start_timestamp'] = pd.concat([pd.Series(['00:00:00.01']), df.stop_timestamp]).values
#     nadf['stop_timestamp'] = pd.concat([df.start_timestamp, pd.Series([f'{datetime.timedelta(seconds=duration)}'])]).values
#     nadf['start_frame'] = pd.concat([pd.Series([0]), df.stop_frame]).values
#     nadf['stop_frame'] = pd.concat([df.start_frame, pd.Series([int(duration * fps)])]).values
#     nadf['narration_timestamp'] = nadf['start_timestamp']
#     nadf1 = nadf[nadf.start_frame + min_size < nadf.stop_frame]
#     return nadf1
#     # return pd.concat([df, nadf1]).sort_values('start_frame')


import datetime
def get_noaction(df, video_info, min_size=0):
    nadf = pd.concat([df, df.iloc[-1:]]).sort_values('start_frame')
    # fix for actions inside other actions
    rows = list(nadf.iterrows())
    for (i, rowm1), (j, row) in zip(rows, rows[1:]):
        if rowm1.stop_frame > row.stop_frame:
            row['stop_frame'] = rowm1.stop_frame
            row['stop_timestamp'] = rowm1.stop_timestamp

    nadf['src_narration_id'] = nadf.narration_id.copy()
    nadf.narration_id.iloc[-1] += '_final'
    nadf['narration_id'] = nadf.narration_id.apply(lambda x: f'{x}_noaction')
    nadf['verb'] = ''
    nadf['noun'] = ''
    nadf['verb_class'] = -1
    nadf['noun_class'] = -1
    nadf['all_nouns'] = nadf.all_nouns.apply(lambda x: [])
    nadf['all_noun_classes'] = nadf.all_nouns.apply(lambda x: [])

    duration = video_info.duration[df.name]
    fps = video_info.fps[df.name]
    nadf['start_timestamp'] = pd.concat([pd.Series(['00:00:00.01']), df.stop_timestamp]).values
    nadf['stop_timestamp'] = pd.concat([df.start_timestamp, pd.Series([f'{datetime.timedelta(seconds=duration)}'])]).values
    nadf['start_frame'] = pd.concat([pd.Series([0]), df.stop_frame]).values
    nadf['stop_frame'] = pd.concat([df.start_frame, pd.Series([int(duration * fps)])]).values
    nadf['narration_timestamp'] = nadf['start_timestamp']
    nadf1 = nadf[nadf.start_frame + min_size < nadf.stop_frame]
    nadf1 = nadf1.drop_duplicates(subset=['narration_id'])
    return nadf1

def add_noaction(df, video_info, min_size=0):
    nadf = get_noaction(df, video_info, min_size)
    return pd.concat([df, nadf]).sort_values('start_frame')



def _load_existing_splits(path):
    # P='^[a-z]+(\d+)/+\w+/(.+)\..*$'
    # df=pd.Series(open(path).read().splitlines())
    # df=df.apply(lambda x: pd.Series(re.match(P, x).groups(), ['verb_class','narration_id']))
    # df['verb_class'] = df.verb_class.astype(int)
    df=pd.read_csv(path, header=0, names=['verb_class', 'narration_id'])
    df['video_id'] = df.narration_id.apply(lambda x: x.rsplit('_', 1)[0])
    return df

def create_epic_kitchens_noaction_splits(ann_dir='epic-kitchens-100-annotations'):
    df = pd.concat([
        pd.read_csv(os.path.join(ann_dir, 'EPIC_100_train.csv')).assign(split='train'),
        pd.read_csv(os.path.join(ann_dir, 'EPIC_100_validation.csv')).assign(split='val')
    ])
    video_info = pd.read_csv(os.path.join(ann_dir, 'EPIC_100_video_info.csv')).set_index('video_id')
    print(df.head())

    train_fname = 'configs/projects/hyrsm/epic_kitchens/train_few_shot.txt'
    test_fname = 'configs/projects/hyrsm/epic_kitchens/test_few_shot.txt'
    refdf_train = _load_existing_splits(train_fname)
    refdf_test = _load_existing_splits(test_fname)
    refdf_train = refdf_train[~refdf_train.narration_id.str.contains('_noaction')]
    refdf_test = refdf_test[~refdf_test.narration_id.str.contains('_noaction')]
    print(len(refdf_train))
    print(len(refdf_test))

    noaction = df.groupby('video_id').apply(get_noaction, video_info=video_info, min_size=8)
    print(noaction.src_narration_id)
    print(refdf_train.narration_id)
    print(refdf_test.narration_id)
    print(noaction['src_narration_id'].isin(refdf_train.narration_id).sum())
    print(noaction['src_narration_id'].isin(refdf_test.narration_id).sum())
    noaction_train = noaction[noaction['src_narration_id'].isin(refdf_train.narration_id)]
    noaction_test = noaction[noaction['src_narration_id'].isin(refdf_test.narration_id)]
    # print(noaction_train.head())
    # print(noaction_test.head())

    COLS = ['verb_class', 'narration_id']
    refdf_train = pd.concat([refdf_train[COLS], noaction_train[COLS].sample(4000)])
    refdf_test = pd.concat([refdf_test[COLS], noaction_test[COLS].sample(4000)])

    # refdf_train['verb_class'][refdf_train.verb_class == -1] = refdf_train.verb_class.max()+1
    # refdf_test['verb_class'][refdf_test.verb_class == -1] = refdf_test.verb_class.max()+1
    print(refdf_train.verb_class.value_counts())
    print(refdf_test.verb_class.value_counts())
    print(refdf_train.head().to_csv(index=False))
    print(refdf_test.head().to_csv(index=False))
    print(refdf_train.tail().to_csv(index=False))
    print(refdf_test.tail().to_csv(index=False))
    refdf_train.to_csv(f'{train_fname.rsplit(".")[0]}_noaction.txt', index=False, header=False)
    refdf_test.to_csv(f'{test_fname.rsplit(".")[0]}_noaction.txt', index=False, header=False)
    # print(refdf_train.head())
    # print(refdf_test.head())


    

# /vast/bs3639/datasets/epic-kitchens-100-annotations
def create_epic_kitchens_splits(ann_dir='epic-kitchens-100-annotations'):
    # 
    df = pd.concat([
        pd.read_csv(os.path.join(ann_dir, 'EPIC_100_train.csv')).assign(split='train'),
        pd.read_csv(os.path.join(ann_dir, 'EPIC_100_validation.csv')).assign(split='val')
    ])
    print("Epic kitchens split size:")
    print(df.split.value_counts())
    print('total', len(df))
    df_verb_cls = pd.read_csv(os.path.join(ann_dir, 'EPIC_100_verb_classes.csv'))
    # norm_verb = {
    #     v: vn
    #     for vn, l in df_verb_cls.set_index('key').instances.items()
    #     for v in eval(l)
    # }
    # df['verb'] = df.verb.apply(lambda x: norm_verb.get(x))
    # train_df = df[df.verb.isin(ek_train_classes)]
    # test_df = df[df.verb.isin(ek_test_classes)]
    # # print(len(train_df), len(test_df), len(df))
    # # print(len(ek_train_classes), len(ek_test_classes), len(set(train_df.verb)), len(set(test_df.verb)), len(set(df.verb)))
    
    # assert len(train_df) + len(test_df) == len(df), (len(train_df), len(test_df), len(df))
    # 
    refdf_train = _load_existing_splits('configs/projects/hyrsm/epic_kitchens/train_few_shot.txt')
    refdf_train['verb'] = refdf_train.verb_class.apply(lambda i: ek_train_classes[i])
    refdf_test = _load_existing_splits('configs/projects/hyrsm/epic_kitchens/test_few_shot.txt')
    refdf_test['verb'] = refdf_test.verb_class.apply(lambda i: ek_test_classes[i])
    print('any duplicates? max count:', refdf_train.narration_id.value_counts().max())
    refdf = pd.concat([refdf_train, refdf_test])
    
    missing = set(df.verb) - (set(refdf_train.verb)|set(refdf_test.verb))
    missing_count = df.verb[df.verb.isin(missing)].value_counts()
    print(len(missing), missing)
    print(len(missing_count))
    print(missing_count)

    print('ref', len(set(refdf.verb)))
    print('df', len(set(df.verb)))
    print('verb ref-df', len(set(refdf.verb) - set(df.verb)))
    print('verb df-ref', len(set(df.verb) - set(refdf.verb)))

    print(len(refdf_train))
    print(len(refdf_test))
    print(len(refdf_train) + len(refdf_test), len(df))
    print(len(set(refdf_train.narration_id) | set(refdf_test.narration_id)), len(set(df.narration_id)))
    print(len(set(df.narration_id) - set(refdf_train.narration_id)))
    print(len(set(refdf_train.narration_id) - set(df.narration_id)))
    print(len(set(refdf_train.narration_id) | set(df.narration_id)))



    # # import ipdb
    # # ipdb.set_trace()

    # s1 = set(train_df.narration_id)
    # s2 = set(refdf_train.narration_id)
    # print(len(s1), len(s2), len(s1-s2), len(s2-s1))

    # # print("all training", set(refdf_train.video_id))
    # # print("all testing", set(refdf_test.video_id))
    # print("overlap", set(refdf_train.video_id) | set(refdf_test.video_id))
    # print("only training", set(refdf_train.video_id) - set(refdf_test.video_id))
    # print("only testing", set(refdf_test.video_id) - set(refdf_train.video_id))

def create_egtea_splits(data_dir, split_dir='configs/projects/hyrsm/egtea'):
    # load all narrations
    df = pd.read_csv(os.path.join(
        data_dir,
        'raw_annotations/action_labels.csv'
    ), delimiter=r';\s*', engine='python').rename({
        'Clip Prefix (Unique)': 'narration_id',
        'Video Session': 'video_id',
        'Starting Time (ms)': 'start_time',
        'Ending Time (ms)': 'stop_time',
        'Action Label': 'action',
        'Verb Label': 'verb',
        'Noun Label(s)': 'nouns',
    }, axis=1).set_index('narration_id')

    idx_df = pd.read_csv(os.path.join(
        data_dir,
        'raw_annotations/cls_label_index.csv'
    ), delimiter=r';\s*', engine='python').rename({
        '# Action ID': 'action_class',
        'Action Label': 'action',
        'Verb Label': 'verb',
        'Noun Label(s)': 'nouns',
    }, axis=1)
    print(idx_df)

    # get verb class index in order of appearance in action list
    verb_class = idx_df.groupby('verb').apply(lambda x: x.action_class.min())
    # include verbs missing from index ???????
    for c in list(set(df.verb.unique()) - set(verb_class.index)):
        print("Missing", c, "adding...")
        verb_class[c] = 1000000
    verb_class = pd.Series(np.arange(len(verb_class)), verb_class.sort_values().index)
    df['verb_class'] = df.verb.apply(lambda v: verb_class[v])

    print(df.verb_class.value_counts())

    print("Verb Index")
    print(verb_class)
    print()

    print(df)
    # configs/projects/hyrsm/egtea/test_few_shot.txt
    f = os.path.join(split_dir, 'test_few_shot.txt')
    df.reset_index()[['verb_class', 'narration_id']].to_csv(f, index=False, header=False)
    print('wrote to', f)

def create_meccano_splits(data_dir, split_dir='configs/projects/hyrsm/meccano'):
    kw=dict(dtype={'video_id': str})
    df = pd.concat([
        pd.read_csv(os.path.join(data_dir, 'MECCANO_train_actions.csv'), **kw).assign(split='Train'),
        pd.read_csv(os.path.join(data_dir, 'MECCANO_test_actions.csv'), **kw).assign(split='Test'),
        pd.read_csv(os.path.join(data_dir, 'MECCANO_val_actions.csv'), **kw).assign(split='Val'),
    ]).rename({'end_frame': 'stop_frame'}, axis=1)
    # convert existing columns
    df['start_frame'] = df.start_frame.apply(lambda x: int(x.split('.')[0]))
    df['stop_frame'] = df.stop_frame.apply(lambda x: int(x.split('.')[0]))
    df['video_id'] = df.apply(lambda x: f'{x.split}/{x.video_id}', axis=1)
    df['narration_id'] = df.apply(lambda x: f'{x.video_id}-{x.start_frame}-{x.stop_frame}', axis=1)
    # convert action to verb
    df['verb'] = df.action_name.apply(lambda x: x.split('_')[0])
    verb_index = sorted(df.verb.unique())
    verb_class = pd.Series(range(len(verb_index)), verb_index)
    df['verb_class'] = df.verb.apply(lambda x: verb_class[x])

    # write verb index
    print(verb_class)
    open(os.path.join(split_dir, 'class_index.txt'), 'w').write('\n'.join(verb_index))

    # write split file
    print(df[['verb_class', 'narration_id']].head().to_csv(index=False, header=False))
    df[['verb_class', 'narration_id']].to_csv(os.path.join(split_dir, 'test_few_shot.txt'), index=False, header=False)



def create_egtea_epic_compat_annotations(data_dir, split_dir='configs/projects/hyrsm/egtea'):
    # narration_id,participant_id,video_id,
    # narration_timestamp,start_timestamp,stop_timestamp,
    # start_frame,stop_frame,
    # narration,verb,verb_class,noun,noun_class,all_nouns,all_noun_classes
    # load all narrations
    df = pd.read_csv(os.path.join(
        data_dir,
        'raw_annotations/action_labels.csv'
    ), delimiter=r';\s*', engine='python').rename({
        'Clip Prefix (Unique)': 'narration_id',
        'Video Session': 'video_id',
        'Starting Time (ms)': 'start_time',
        'Ending Time (ms)': 'stop_time',
        'Action Label': 'action',
        'Verb Label': 'verb',
        'Noun Label(s)': 'all_nouns',
    }, axis=1).set_index('narration_id')

    idx_df = pd.read_csv(os.path.join(
        data_dir,
        'raw_annotations/cls_label_index.csv'
    ), delimiter=r';\s*', engine='python').rename({
        '# Action ID': 'action_class',
        'Action Label': 'action',
        'Verb Label': 'verb',
        'Noun Label(s)': 'nouns',
    }, axis=1)
    print(idx_df)

    verb_index = open(os.path.join(split_dir, 'class_index.txt')).read().splitlines()   
    inv_verb_index = pd.Series(range(len(verb_index)), verb_index) 
    ek_verb_index = open(os.path.join(split_dir, 'ek_class_index.txt')).read().splitlines()
    ek_verb_index = {i: v for i, v in enumerate(ek_verb_index) if v}

    df['verb_class'] = df.verb.apply(lambda v: inv_verb_index[v])
    df['verb'] = df.verb_class.apply(lambda v: ek_verb_index[v])

    # TODO: nouns

    # df['action'] = df.apply(lambda )

    # get verb class index in order of appearance in action list
    verb_class = idx_df.groupby('verb').apply(lambda x: x.action_class.min())
    # include verbs missing from index ???????
    for c in list(set(df.verb.unique()) - set(verb_class.index)):
        print("Missing", c, "adding...")
        verb_class[c] = 1000000
    verb_class = pd.Series(np.arange(len(verb_class)), verb_class.sort_values().index)
    df['verb_class'] = df.verb.apply(lambda v: verb_class[v])

    print(df.verb_class.value_counts())

    print("Verb Index")
    print(verb_class)
    print()



if __name__ == '__main__':
    import fire 
    fire.Fire()