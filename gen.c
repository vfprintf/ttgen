/*
 * TODO: teamk kun perfk=2 on vähän rikki parittomilla määrillä joukkueita
 */

#ifndef DEBUG
#define NDEBUG
#endif

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <alloca.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#define DIVCEIL(p, q) (((p) + (q) - 1) / (q))
#define CEIL2(p) ((p) + (p)%2)

#ifdef DEBUG
#define DPRINTF(f, ...) fprintf(stderr, f, __VA_ARGS__)
#define DPRINT(x) fprintf(stderr, "%s", x)
#else
#define DPRINTF(...) ((void)0)
#define DPRINT(x) ((void)0)
#endif

struct imat8 {
	int rows;
	int cols;
	uint8_t data[];
};

struct imat64 {
	int rows;
	int cols;
	uint64_t data[];
};

struct mdelta {
	uint16_t a;
	uint16_t b;
	int32_t c;
};

#define MGET(m, i, j) ((m)->data[(i)*(m)->cols+(j)])
#define MSET(m, i, j, e) do{ (m)->data[(i)*(m)->cols+(j)] = (e); }while(0)
#define MROW(m, i) (&(m)->data[(i)*(m)->cols])

#define PARSE_MODE 1
#define INIT_MODE  2
#define TEAM_MODE  4
#define JUDGE_MODE 8
//#define DELTA_MODE 16

//#define EMPTY8     0xff
static const uint8_t EMPTY8 = 0xff;
//#define CANCEL     0xffffffff
static const int32_t CANCEL = 0xffffffff;

struct {
	int flags;
} config = {
	.flags = 0
};

struct genparms {
	struct imat8 *cvec;
	int *bsizes;
	int bcount;
	int elem_count;
	int arenas;
	int perfk, teamk;
	int ccnt;
	int height;
};

static int mdcmp(const void *a, const void *b);
static size_t ncr(ssize_t n, ssize_t k);
static void shuffle(void *p, size_t num, size_t size);
static int max(int a, int b);
static int min(int a, int b);
static void memswap(void *a, void *b, size_t size);
static void memswap_small(void *a, void *b, int bits);
static struct imat8 *i8alloc(int rows, int cols);
static struct imat64 *i64alloc(int rows, int cols);
static void i8zero(struct imat8 *m);
static void i64zero(struct imat64 *m);
static void i8init(struct imat8 *m, int rows, int cols);
static void i8print(struct imat8 *m, FILE *f);
static void i8set_dyn(struct imat8 **m, int row, int col, int e);
static void i8apply(struct imat8 *dest, struct mdelta delta);
static void i8apply_all(struct imat8 **dest, struct mdelta delta, int cnt);
static uint64_t i8multimask(struct imat8 **mats, int cnt, int row, int col);
static int rowcv(struct imat64 *cmat, struct imat8 **mats, int cnt, int row, uint64_t *mask);
static int pow2(int x);

struct team_block {
	int perfcount;
	int perf_s;
	// teams = [start, end[
	int start;
	int end;
	int iter;
	int gen;
};

struct runparms {
	void (*init)(struct runparms *, struct genparms *);
	int32_t (*cost_delta)(struct runparms *, struct mdelta);
	int (*validate)(struct runparms *, struct genparms *);

	struct imat8 **mats;
	struct imat8 **run_mats;
	int mat_cnt;
	int pop_cnt;
	int e_cnt;
	int cv_cnt;

	struct imat8 *smat_up, *smat_down;
	struct imat64 *cmat;

	int w_constv;
	int w_lastrow_sparse;
	int w_streak_diff;
	int w_streak_switch;

	int s_converge_iterations;
	int s_tabu_iterations;
	int s_min_tabu_iterations;

	int c_streak_target;
	int c_empty_min;
};

#define S_MANY	0x80
#define S_EMPTY	0x40
#define S_CMASK 0x3f

static void rp_init(struct runparms *rp, struct genparms *gp);
static void rp_alloc_mats(struct runparms *rp, struct genparms *gp, int rows, int cols, int cnt);
static void rp_free(struct runparms *rp);
static void rp_apply(struct runparms *rp, struct mdelta delta);
static void rp_apply_remove(struct runparms *rp, int row, uint8_t e);
static void rp_apply_add(struct runparms *rp, int row, uint8_t e);
static void rp_combine_ud(struct runparms *rp, int row, uint8_t e, uint8_t flag);
static int rp_run_trim(struct runparms *rp);
static void rp_begin(struct runparms *rp);
static void rp_restreak(struct runparms *rp);
static int rp_calc_cvs(struct runparms *rp);
static void rp_ser(struct runparms *rp, struct genparms *gp, FILE *fp);

static void team_init(struct runparms *rp, struct genparms *gp);
static int team_calc_t_upper_bound(struct team_block *blocks, struct genparms *gp);
static void team_block_init(struct team_block *p, struct genparms *gp, int *block_P, int *block_S);
static void team_gen_idx_k2(uint8_t *idx0, uint8_t *idx1, struct team_block *b);
static void team_gen_idx_k1(uint8_t *idx, struct team_block *b);
static int team_block_cmp(const void *a, const void *b);
static int32_t team_cost_delta(struct runparms *rp, struct mdelta delta);
static int team_mat_validate(struct runparms *rp, struct genparms *gp);

static void judge_init(struct runparms *rp, struct genparms *gp);
static int32_t judge_cost_delta(struct runparms *rp, struct mdelta delta);
static int judge_validate(struct runparms *rp, struct genparms *gp);

static int32_t cost_constv(struct runparms *rp, int from, int to, uint8_t e);
static int32_t cost_move(struct runparms *rp, int from, int to, uint8_t e, uint8_t flag, int32_t (*S)(uint8_t, uint8_t), uint8_t l);
static void near_streaks(struct runparms *rp, uint8_t *p, uint8_t *n, int row, uint8_t e, uint8_t flag);
static int32_t streak_target(uint8_t x, uint8_t lambda);
static int32_t streak_decay(uint8_t x, uint8_t lambda);
static int32_t cost_switch(struct runparms *rp, int from, int to, uint8_t e);

static void tabu_search(struct runparms *rp);
static int tabu_find_solutions(struct mdelta *dst, struct runparms *rp);

/*
 * gen.c parametrit ([J] = Joukkuetila, [T] = Tuomaritila):
 * 
 * Yleiset parametrit:
 *   -t                      Aja ohjelma joukkuetilassa
 *   -j                      Aja ohjelma tuomaritilassa
 *   -M <määrä>              [J] joukkeidein määrä. [T] tuomarien määrä
 *   -a <kenttien määrä>     Käytettävissä olevien kenttien määrä
 *   -I                      Lopeta alustusfunktion jälkeen
 *   -C <rivi,sarake,id|x>*  Lisää paikkaa ja id:tä koskeva rajoite
 *   -R <siemen>             Satunnaislukusiemen
 *   
 * [J] Joukkuetilan parametrit:
 *   -b <lohkojen määrä>     Moneenko lohkoon joukkueet jaetaan (anna ENNEN -B parametrejä)
 *   -B <lohkon koko>*       YHDEN lohkon koko (näitä pitää antaa täsmälleen -b kappaletta)
 *   -k <perfk>              perfk parametri (joukkeita per suoritus)
 *   -g <teamk>              teamk parametri (suorituksia per joukkue)
 *
 * [T] Tuomaritilan parametrit:
 *   -h <timeslots>          Montako aikapaikkaa
 *
 * Tabuhaun parametrit:
 *    -i <paino>             Rajoitteen rikkomisen painoarvo hintafunktiossa
 *    -d <paino>             Putkeen olevien suoritusten painoarvo hintafunktiossa
 *    -s <paino>             Viimeisen rivin tyhjyyden painoarvo
 *    -w <paino>             Kentän vaihdoksen painoarvo
 *    -c <määrä>             Iteraatioiden määrä, jonka jälkeen jos kehitystä ei ole tapahtunu haku loppuu
 *    -T <määrä>             Iteraatioiden määrä, jolloin delta on tabu
 *    -S <määrä>             Haluttu putkeen olevien suoritusten määrä
 *    -E <määrä>             Haluttu (minimi) tauon pituus
 *
 *
 * Palautusarvot:
 *   0 - ok
 *   1 - väärä tila
 *   2 - väärä parametri
 *   3 - väärä määrä joukkueita/tuomareita 
 */
int main(int argc, char **argv){
	struct genparms gp = {
		.bsizes = NULL,
		.bcount = 1,
		.perfk = 1,
		.teamk = 0,
		.ccnt = 0,
		.cvec = NULL
	};

	struct runparms rp = {
		.w_constv = 0,
		.w_lastrow_sparse = 0,
		.w_streak_diff = 2,
		.w_streak_switch = 1,
		.s_converge_iterations = 1000,
		.s_tabu_iterations = 20,
		.s_min_tabu_iterations = 0,
		.c_streak_target = 3,
		.c_empty_min = 2
	};

	int opt, bind = 0;
	unsigned int seed = (unsigned int) time(NULL);
	while((opt = getopt(argc, argv, "tjIPR:M:a:b:B:k:g:h:i:d:s:w:c:T:S:E:C:")) != -1){
		switch(opt){
			case 't': config.flags |= TEAM_MODE; break;
			case 'j': config.flags |= JUDGE_MODE; break;
			case 'I': config.flags |= INIT_MODE; break;
			case 'P': config.flags |= PARSE_MODE; break;
			case 'R': seed = (unsigned int) atoi(optarg); break;
			case 'M': gp.elem_count = atoi(optarg); break;
			case 'a': gp.arenas = atoi(optarg); break;
			case 'b': gp.bcount = atoi(optarg); gp.bsizes = alloca(gp.bcount * sizeof(int)); break;
			case 'B': gp.bsizes[bind++] = atoi(optarg); break;
			case 'k': gp.perfk = atoi(optarg); break;
			case 'g': gp.teamk = atoi(optarg); break;
			case 'h': gp.height = atoi(optarg); break;
			case 'i': rp.w_constv = atoi(optarg); break;
			case 'd': rp.w_streak_diff = atoi(optarg); break;
			case 's': rp.w_lastrow_sparse = atoi(optarg); break;
			case 'w': rp.w_streak_switch = atoi(optarg); break;
			case 'c': rp.s_converge_iterations = atoi(optarg); break;
			case 'T': rp.s_tabu_iterations = atoi(optarg); break;
			case 'S': rp.c_streak_target = atoi(optarg); break;
			case 'E': rp.c_empty_min = atoi(optarg); break;
			case 'C':
				if(!gp.cvec)
					gp.cvec = i8alloc(8, 3);
				char *pch = strtok(optarg, ",");
				for(int i=0;i<3;i++){
					if(pch == NULL)
						break;
					i8set_dyn(&gp.cvec, gp.ccnt, i, *pch >= '0' && *pch <= '9' ? atoi(pch) : EMPTY8);
					pch = strtok(NULL, ",");
				}
				gp.ccnt++;
				break;
			default: return 2;
		}
	}

	if((!!(config.flags & TEAM_MODE)) == (!!(config.flags & JUDGE_MODE)))
		return 1;

	if(gp.elem_count > 64)
		return 3;
	if(gp.arenas > 256 || gp.height > 256)
		return 3;

	if(!rp.w_lastrow_sparse)
		rp.w_lastrow_sparse = gp.elem_count * gp.arenas * 100;
	if(!rp.w_constv)
		rp.w_constv = 5 * rp.w_lastrow_sparse;

	DPRINTF("Random seed ... %d\n", seed);
	srand(seed);

	DPRINT("Calculating init matrices ... ");
	rp_init(&rp, &gp);
	DPRINTF("%d * %dx%d\n", rp.mat_cnt, rp.mats[0]->rows, rp.mats[0]->cols);

	if(gp.cvec)
		free(gp.cvec);

	rp_begin(&rp);

	if(!(config.flags & INIT_MODE)){
		tabu_search(&rp);
	}

#ifdef DEBUG
	DPRINT("Upmatrix:\n");
	i8print(rp.smat_up, stderr);
	DPRINT("\nDownmatrix:\n");
	i8print(rp.smat_down, stderr);
	DPRINT("\n");
#endif

	rp_ser(&rp, &gp, stdout);

#ifdef DEBUG
	DPRINT("Validating matrices...");
	DPRINTF(" %s\n", rp.validate(&rp, &gp) ? "FAIL" : "OK");
#endif

	rp_free(&rp);

	return 0;
}

static void rp_init(struct runparms *rp, struct genparms *gp){
	if(config.flags & TEAM_MODE){
		rp->init = team_init;
		rp->cost_delta = team_cost_delta;
		rp->validate = team_mat_validate;
	}else if(config.flags & JUDGE_MODE){
		rp->init = judge_init;
		rp->cost_delta = judge_cost_delta;
		rp->validate = judge_validate;
	}

	rp->e_cnt = gp->elem_count;
	rp->init(rp, gp);
	assert(rp->mats);
}

static void rp_alloc_mats(struct runparms *rp, struct genparms *gp, int rows, int cols, int cnt){
	rp->mat_cnt = cnt;
	size_t offset = 2 * rp->mat_cnt * sizeof(struct imat8 *);
	size_t mat_size = sizeof(struct imat8) + rows*cols*sizeof(uint8_t);
	rp->mats = malloc(offset + 2 * rp->mat_cnt * mat_size);
	for(int i=0;i<rp->mat_cnt;i++){
		rp->mats[i] = (struct imat8 *) (((char *) rp->mats) + offset + i*mat_size);
		i8init(rp->mats[i], rows, cols);
	}

	rp->run_mats = rp->mats + rp->mat_cnt;
	for(int i=0;i<rp->mat_cnt;i++){
		rp->run_mats[i] = (struct imat8 *) (((char *) rp->mats) + offset + (i+rp->mat_cnt)*mat_size);
		i8init(rp->run_mats[i], rows, cols);
	}

	size_t smat_size = sizeof(struct imat8) + rows*gp->elem_count*sizeof(uint8_t);
	rp->smat_up = malloc(2*smat_size);
	rp->smat_down = (struct imat8 *) (((char *) rp->smat_up) + smat_size);
	i8init(rp->smat_up, rows, gp->elem_count);
	i8init(rp->smat_down, rows, gp->elem_count);

	rp->cmat = i64alloc(rows, cols);
	i64zero(rp->cmat);

	for(int i=0;i<gp->ccnt;i++){
		int t = MGET(gp->cvec, i, 2);
		assert(t < 64 || t == 0xff);
		int r = MGET(gp->cvec, i, 0);
		int c = MGET(gp->cvec, i, 1);

		MSET(rp->cmat, r, c, MGET(rp->cmat, r, c) | (t != EMPTY8 ? (1ULL << t) : 0xffffffffffffffff));
	}

}

static void rp_begin(struct runparms *rp){
	assert(rp->run_mats > rp->mats);
	rp->cv_cnt = rp_calc_cvs(rp);

	memcpy(*rp->run_mats, *rp->mats, *((char **)rp->run_mats) - *((char **)rp->mats));
	rp_restreak(rp);
}

static int rp_calc_cvs(struct runparms *rp){
	int ret = 0;

	for(int i=0;i<(*rp->mats)->rows;i++){
		ret += rowcv(rp->cmat, rp->mats, rp->mat_cnt, i, NULL);
	}
	
	return ret;
}

static void rp_restreak(struct runparms *rp){
	assert((*rp->run_mats)->rows == rp->smat_up->rows &&
		rp->smat_up->rows == rp->smat_down->rows && rp->smat_up->cols == rp->smat_down->cols);

	i8zero(rp->smat_up);
	i8zero(rp->smat_down);

	int m = 1, t = (*rp->run_mats)->rows, i = 0;
	do{
		uint8_t *lastrow = NULL;

		for(;i!=t;i+=m){
			uint8_t *row = MROW(m > 0 ? rp->smat_up : rp->smat_down, i);
			uint64_t mask = 0, many = 0;

			for(int k=0;k<rp->mat_cnt;k++){
				for(int j=0;j<rp->run_mats[k]->cols;j++){
					uint8_t d = MGET(rp->run_mats[k], i, j);
					if(d == EMPTY8)
						continue;
					if(mask & (1ULL << d))
						many |= 1ULL << d;
					else
						mask |= 1ULL << d;
				}
			}

			for(int j=0;j<rp->smat_up->cols;j++){
				if(lastrow && (!!(lastrow[j] & S_EMPTY) == !(mask & (1ULL << j)))){
					row[j] = (lastrow[j] & S_CMASK) + 1;
				}else{
					row[j] = 0;
				}

				if(!(mask & (1ULL << j)))
					row[j] |= S_EMPTY;
				else if(many & (1ULL << j))
					row[j] |= S_MANY;
			}

			lastrow = row;
		}

		m *= -1;
		i = t - 1;
		t = -1;
	}while(m != 1);

}

static void rp_free(struct runparms *rp){
	free(rp->mats); // frees also rp->run_mats
	free(rp->smat_up); // frees also rp->smat_down
	free(rp->cmat);
}

static void tabu_search(struct runparms *rp){
	DPRINT("Starting tabu search ... ");

	int rows = rp->mats[0]->rows;
	int cols = rp->mats[0]->cols;
	int max_deltas = ncr(rows * cols, 2);
	struct mdelta *deltas = malloc(sizeof(struct mdelta) * max_deltas);

	struct imat64 *tabus = i64alloc(rows * cols, rows * cols);
	i64zero(tabus);

	DPRINTF("%d deltas, %dx%d tabus, %d matrices\n", max_deltas, tabus->rows, tabus->cols, rp->mat_cnt);

	int32_t best_cost = 1 << 30;
	int32_t cur_cost = best_cost;
	int last_prog = 0;

	for(int iter=0;iter<last_prog+rp->s_converge_iterations;iter++){
		DPRINTF("* %d / %d ::", iter+1, last_prog+rp->s_converge_iterations);

		int num_solutions = tabu_find_solutions(deltas, rp);
		DPRINTF(" %d ", num_solutions);
		assert(num_solutions <= max_deltas);
		shuffle(deltas, num_solutions, sizeof(struct mdelta));
		DPRINT(":");
		qsort(deltas, num_solutions, sizeof(struct mdelta), mdcmp);
		DPRINT(": ");

		for(int i=0;i<num_solutions;i++){
			struct mdelta d = deltas[i];
			int tt = MGET(tabus, d.a, d.b);
			int32_t new_cost = cur_cost + d.c;
			if(tt <= iter || (new_cost<best_cost)){
				DPRINTF("%d <--> %d (%s%d) ", d.a, d.b, d.c>0?"+":(d.c==0?"+-":""), d.c);
				assert(d.a < tabus->rows && d.b < tabus->cols);
				MSET(tabus, d.a, d.b, iter+rp->s_tabu_iterations);
				rp_apply(rp, d);
#ifdef DEBUG
				int32_t back_cost = rp->cost_delta(rp, d);
				if(back_cost != -d.c){
					DPRINTF("E: Nonsymmetric cost delta: %d <> %d\n", back_cost, d.c);
					assert(0);
				}
#endif
				assert(rp->cv_cnt >= 0);
				if(rp->cv_cnt == 0 && rp_run_trim(rp)){
					DPRINTF("[TRIM %d] ", (*rp->run_mats)->rows);
				}
				cur_cost = new_cost;
				if(cur_cost < best_cost){
					memcpy(*rp->mats, *rp->run_mats, *((char**)rp->run_mats) - *((char**)rp->mats));
					best_cost = cur_cost;
					last_prog = iter;
				}
				break;
			}
			//else{
			//	DPRINTF("[%d]", d.c);
			//}
		}

		DPRINTF(":: C: %d | B: %d\n", cur_cost, best_cost);
#ifdef DEBUG
		if(last_prog == iter)
			i8print(*rp->run_mats, stderr);
#endif
	}

	free(deltas);
	free(tabus);
}

static int tabu_find_solutions(struct mdelta *dst, struct runparms *rp){
	int len = (*rp->run_mats)->rows * (*rp->run_mats)->cols;
	int ret = 0;

	for(int i=0;i<len;i++){
		for(int j=i+1;j<len;j++){
			for(int k=0;k<rp->mat_cnt;k++){
				if(rp->run_mats[k]->data[i] != rp->run_mats[k]->data[j]){
					dst->a = i;
					dst->b = j;
					int32_t c = rp->cost_delta(rp, *dst);
					if(c != CANCEL){	
						dst->c = c;
						ret++;
						dst++;
					}
					break;
				}
			}
		}
	}

	return ret;
}

static void rp_apply(struct runparms *rp, struct mdelta delta){
	for(int k=0;k<rp->mat_cnt;k++){
		rp->cv_cnt += cost_constv(rp, delta.a, delta.b, rp->run_mats[k]->data[delta.a]);
		rp->cv_cnt += cost_constv(rp, delta.b, delta.a, rp->run_mats[k]->data[delta.b]);
	}

	int row_a = delta.a / (*rp->run_mats)->cols;
	int row_b = delta.b / (*rp->run_mats)->cols;

	if(row_a != row_b){
		for(int k=0;k<rp->mat_cnt;k++){
			rp_apply_remove(rp, row_a, rp->run_mats[k]->data[delta.a]);
			rp_apply_remove(rp, row_b, rp->run_mats[k]->data[delta.b]);
		}
		for(int k=0;k<rp->mat_cnt;k++){
			rp_apply_add(rp, row_a, rp->run_mats[k]->data[delta.b]);
			rp_apply_add(rp, row_b, rp->run_mats[k]->data[delta.a]);
		}
	}

	i8apply_all(rp->run_mats, delta, rp->mat_cnt);
}

static void rp_apply_remove(struct runparms *rp, int row, uint8_t e){
	if(e == EMPTY8)
		return;

	uint8_t sflags = MGET(rp->smat_up, row, e) & ~S_CMASK;
	assert(!(sflags & S_EMPTY));

	if(sflags & S_MANY){
		sflags &= ~S_MANY;
		int ecnt = 0;
		for(int k=0;k<rp->mat_cnt;k++){
			for(int j=0;j<rp->run_mats[k]->cols;j++){
				if((MGET(rp->run_mats[k], row, j) == e) && (++ecnt == 3)){
					sflags |= S_MANY;
					goto __exit_loop__;
				}
			}
		}
__exit_loop__: ((void)0);
	}else{
		sflags |= S_EMPTY;
	}

	MSET(rp->smat_up, row, e, (MGET(rp->smat_up, row, e) & S_CMASK) | sflags);
	MSET(rp->smat_down, row, e, (MGET(rp->smat_down, row, e) & S_CMASK) | sflags);

	if(sflags & S_EMPTY)
		rp_combine_ud(rp, row, e, S_EMPTY);
}

static void rp_apply_add(struct runparms *rp, int row, uint8_t e){
	if(e == EMPTY8)
		return;

	uint8_t dflags = MGET(rp->smat_up, row, e) & ~S_CMASK;

	if(!(dflags & S_EMPTY))
		dflags |= S_MANY;
	else
		dflags = 0;

	MSET(rp->smat_up, row, e, (MGET(rp->smat_up, row, e) & S_CMASK) | dflags);
	MSET(rp->smat_down, row, e, (MGET(rp->smat_down, row, e) & S_CMASK) | dflags);

	if(!dflags)
		rp_combine_ud(rp, row, e, 0);
}

static void rp_combine_ud(struct runparms *rp, int row, uint8_t e, uint8_t flag){
	uint8_t streak;
	if(row > 0 && (MGET(rp->smat_up, row-1, e) & S_EMPTY) == flag){
		streak = (MGET(rp->smat_up, row-1, e) & S_CMASK) + 1;
	}else{
		streak = 0;
	}

	MSET(rp->smat_up, row, e, ((MGET(rp->smat_up, row, e) & ~S_CMASK) | streak));
	if(row+1<rp->smat_up->rows){
		uint8_t f = MGET(rp->smat_up, row+1, e) & S_EMPTY;
		streak = f == flag ? (streak + 1) : 0;
		for(int i=row+1;i<rp->smat_up->rows;i++,streak++){
			uint8_t s = MGET(rp->smat_up, i, e);
			if((s & S_EMPTY) != f)
				break;
			MSET(rp->smat_up, i, e, (s & ~S_CMASK) | streak);
		}
	}

	if(row < rp->smat_down->rows-1 && (MGET(rp->smat_down, row+1, e) & S_EMPTY) == flag){
		streak = (MGET(rp->smat_down, row+1, e) & S_CMASK) + 1;
	}else{
		streak = 0;
	}

	MSET(rp->smat_down, row, e, ((MGET(rp->smat_down, row, e) & ~S_CMASK) | streak));
	if(row > 0){
		uint8_t f = MGET(rp->smat_down, row-1, e) & S_EMPTY;
		streak = f == flag ? (streak + 1) : 0;
		for(int i=row-1;i>=0;i--,streak++){
			uint8_t s = MGET(rp->smat_down, i, e) & S_EMPTY;
			if((s & S_EMPTY) != f)
				break;
			MSET(rp->smat_down, i, e, (s & ~S_CMASK) | streak);
		}
	}
}

static int rp_run_trim(struct runparms *rp){
	// Teoreettinen minimi
	if((*rp->run_mats)->rows == DIVCEIL(rp->pop_cnt, (*rp->run_mats)->cols))
		return 0;

	uint64_t mask;
	rowcv(NULL, rp->run_mats, rp->mat_cnt, (*rp->run_mats)->rows-1, &mask);
	if(!mask){
		for(int i=0;i<rp->mat_cnt;i++)
			rp->run_mats[i]->rows--;
		return 1;
	}
	return 0;
}

static int mdcmp(const void *a, const void *b){
	return ((struct mdelta *)a)->c - ((struct mdelta *)b)->c;
}

static struct imat8 *i8alloc(int rows, int cols){
	struct imat8 *ret = malloc(sizeof(struct imat8) + rows * cols * sizeof(uint8_t));
	i8init(ret, rows, cols);
	return ret;
}

static struct imat64 *i64alloc(int rows, int cols){
	struct imat64 *ret = malloc(sizeof(struct imat64) + rows * cols * sizeof(uint64_t));
	i8init((struct imat8 *) ret, rows, cols);
	return ret;
}

static void i8zero(struct imat8 *m){
	memset(&m->data, 0, m->rows * m->cols * sizeof(uint8_t));
}

static void i64zero(struct imat64 *m){
	memset(&m->data, 0, m->rows * m->cols * sizeof(uint64_t));
}

static void i8init(struct imat8 *m, int rows, int cols){
	m->rows = rows;
	m->cols = cols;
}

static void i8set_dyn(struct imat8 **m, int row, int col, int e){
	if(row >= (*m)->rows){
		(*m)->rows = (*m)->rows * 2;
		*m = realloc(*m, sizeof(struct imat8) + (*m)->rows * (*m)->cols * sizeof(uint8_t));
		if(!*m){
			fprintf(stderr, "Voi paska realloc epäonnistui\n");
			return;
		}
	}
	MSET(*m, row, col, e);
}

static void i8apply(struct imat8 *dest, struct mdelta delta){
	assert(delta.a < dest->rows*dest->cols && delta.b < dest->rows*dest->cols);

	int t = dest->data[delta.a];
	dest->data[delta.a] = dest->data[delta.b];
	dest->data[delta.b] = t;
}

static void i8apply_all(struct imat8 **dest, struct mdelta delta, int cnt){
	for(int i=0;i<cnt;i++,dest++){
		i8apply(*dest, delta);
	}
}

static uint64_t i8multimask(struct imat8 **mats, int cnt, int row, int col){
	int idx = (*mats)->cols * row + col;
	uint64_t ret = 0;
	for(int i=0;i<cnt;i++){
		if(mats[i]->data[idx] != EMPTY8)
			ret |= 1ULL << mats[i]->data[idx];
	}
	return ret;
}

static void i8print(struct imat8 *m, FILE *f){
	fprintf(f, "     ");
	for(int i=0;i<m->cols;i++){
		fprintf(f, "%-3d ", i);
	}
	fprintf(f, "\n");
	for(int i=0;i<m->cols+1;i++){
		fprintf(f, "----");
	}
	fprintf(f, "\n");
	for(int i=0;i<m->rows;i++){
		fprintf(f, "%-3d| ", i);
		for(int j=0;j<m->cols;j++){
			fprintf(f, "%-3d ", MGET(m, i, j));
		}
		fprintf(f, "\n");
	}
}

static void rp_ser(struct runparms *rp, struct genparms *gp, FILE *fp){
	for(int i=0;i<(*rp->mats)->rows;i++){
		for(int j=0;j<(*rp->mats)->cols;j++){
			int e = MGET(rp->mats[0], i, j);
			int block = 0;
			if(gp->bsizes){
				for(int k=gp->bsizes[0];k<=e;k+=gp->bsizes[++block]);
			}
			fprintf(fp, "%d\t%d\t%d", i, block, j);
			for(int k=0;k<rp->mat_cnt;k++){
				fprintf(fp, "\t%d", MGET(rp->mats[k], i, j));
			}
			fprintf(fp, "\n");
		}
	}
}

static void shuffle(void *p, size_t num, size_t size){
	char *base = (char *) p;

	for(size_t i=0;i<num;i++){
		unsigned int idx = rand() % num;
		if(i != idx)
			memswap(base + i*size, base + idx*size, size);
	}
}

static size_t ncr(ssize_t n, ssize_t k){
	if(k < 0 || n < 0)
		return 0;
	size_t ret = 1;
	for(int i=1;i<=k;i++){
		ret *= n - (k - i);
		ret /= i;
	}
	return ret;
}

static int max(int a, int b){
	return a > b ? a : b;
}

static int min(int a, int b){
	return a < b ? a : b;
}

static void memswap(void *a, void *b, size_t size){
	char tmp[size];
	memcpy(tmp, a, size);
	memcpy(a, b, size);
	memcpy(b, tmp, size);
}

static void memswap_small(void *a, void *b, int bits){
	uint64_t mask = (1ULL << bits) - 1;
	uint64_t tmp = (*((uint64_t *)a)) & mask;
	*((uint64_t *)a) = (*((uint64_t *)a) & ~mask) | ((*((uint64_t *)b)) & mask);
	*((uint64_t *)b) = (*((uint64_t *)b) & ~mask) | tmp;
}

static int pow2(int x){
	return x * x;
}

static int32_t team_cost_delta(struct runparms *rp, struct mdelta delta){
	assert(delta.b > delta.a);
	assert((*rp->run_mats)->data[delta.a] != EMPTY8 || (*rp->run_mats)->data[delta.b] != EMPTY8);

	//if(cost_constv(rp, delta))
	//	return CANCEL;

	int row_a = delta.a / (*rp->run_mats)->cols;
	int row_b = delta.b / (*rp->run_mats)->cols;

	//if(row_a == row_b)
	//	return 0;

	int32_t ret = 0;

	for(int i=0;i<rp->mat_cnt;i++){
		uint8_t ea = rp->run_mats[i]->data[delta.a];
		uint8_t eb = rp->run_mats[i]->data[delta.b];
		if(ea != EMPTY8){
			ret += rp->w_constv * cost_constv(rp, delta.a, delta.b, ea);
			ret += rp->w_streak_diff *
				cost_move(rp, delta.a, delta.b, ea, 0, streak_target, rp->c_streak_target);
			ret += rp->w_streak_diff *
				cost_move(rp, delta.b, delta.a, ea, S_EMPTY, streak_decay, rp->c_empty_min);
			ret += rp->w_streak_switch * cost_switch(rp, delta.a, delta.b, ea);
		}
		if(eb != EMPTY8){
			ret += rp->w_constv * cost_constv(rp, delta.b, delta.a, eb);
			ret += rp->w_streak_diff *
				cost_move(rp, delta.b, delta.a, eb, 0, streak_target, rp->c_streak_target);
			ret += rp->w_streak_diff *
				cost_move(rp, delta.a, delta.b, eb, S_EMPTY, streak_decay, rp->c_empty_min);
			ret += rp->w_streak_switch * cost_switch(rp, delta.b, delta.a, eb);
		}
	}

	if(row_a != row_b && row_b == (*rp->run_mats)->rows-1){
		if((*rp->run_mats)->data[delta.a] == EMPTY8){
			ret -= rp->w_lastrow_sparse;
		}else if((*rp->run_mats)->data[delta.b] == EMPTY8){
			ret += rp->w_lastrow_sparse;
		}
	}

	return ret;
}

static int32_t judge_cost_delta(struct runparms *rp, struct mdelta delta){
	int32_t ret = 0;

	uint8_t ea = (*rp->run_mats)->data[delta.a];
	uint8_t eb = (*rp->run_mats)->data[delta.b];

	if(ea == EMPTY8 || eb == EMPTY8)
		return CANCEL;

	ret += rp->w_constv * cost_constv(rp, delta.a, delta.b, ea);
	ret += rp->w_constv * cost_constv(rp, delta.b, delta.a, eb);
	ret += rp->w_streak_diff * cost_move(rp, delta.a, delta.b, ea, 0, streak_target, rp->c_streak_target);
	ret += rp->w_streak_diff * cost_move(rp, delta.b, delta.a, ea, S_EMPTY, streak_decay, rp->c_empty_min);
	ret += rp->w_streak_diff * cost_move(rp, delta.b, delta.a, eb, 0, streak_target, rp->c_streak_target);
	ret += rp->w_streak_diff * cost_move(rp, delta.a, delta.b, eb, S_EMPTY, streak_decay, rp->c_empty_min);
	ret += rp->w_streak_switch * cost_switch(rp, delta.a, delta.b, ea);
	ret += rp->w_streak_switch * cost_switch(rp, delta.b, delta.a, eb);

	return ret;
}

static int32_t cost_switch(struct runparms *rp, int from, int to, uint8_t e){
	int row_src = from / (*rp->run_mats)->cols;
	int col_src = from % (*rp->run_mats)->cols;
	int row_dest = to / (*rp->run_mats)->cols;
	int col_dest = to % (*rp->run_mats)->cols;

	uint64_t ebit = 1ULL << e;

	if(i8multimask(rp->run_mats, rp->mat_cnt, row_dest, col_dest) & ebit)
		return 0;

	int32_t ret = 0;

	if(row_src > 0 &&  !(MGET(rp->smat_up, row_src-1, e) & S_EMPTY) 
		&& !(i8multimask(rp->run_mats, rp->mat_cnt, row_src-1, col_src) & ebit))
		ret--;

	if(row_src < (*rp->run_mats)->rows-1 && !(MGET(rp->smat_up, row_src+1, e) & S_EMPTY) 
		&& !(i8multimask(rp->run_mats, rp->mat_cnt, row_src+1, col_src) & ebit))
		ret--;

	if(row_dest > 0 && (row_src != row_dest-1 ?
		(!(MGET(rp->smat_up, row_dest-1, e) & S_EMPTY) && 
		!(i8multimask(rp->run_mats, rp->mat_cnt, row_dest-1, col_dest) & ebit)) :
		((MGET(rp->smat_up, row_src, e) & S_MANY) && (col_src == col_dest ||
		!(i8multimask(rp->run_mats, rp->mat_cnt, row_src, col_dest) & ebit)))))
		ret++;

	if(row_dest < (*rp->run_mats)->rows-1 && (row_src != row_dest+1 ?
		(!(MGET(rp->smat_up, row_dest+1, e) & S_EMPTY) &&
		!(i8multimask(rp->run_mats, rp->mat_cnt, row_dest+1, col_dest) & ebit)) :
		((MGET(rp->smat_up, row_src, e) & S_MANY) && (col_src == col_dest ||
		!(i8multimask(rp->run_mats, rp->mat_cnt, row_src, col_dest) & ebit)))))
		ret++;

	return ret;
}

static int32_t cost_move(struct runparms *rp, int from, int to, uint8_t e, uint8_t flag, int32_t (*S)(uint8_t, uint8_t), uint8_t l){
	if(e == EMPTY8)
		return 0;

	int row_src = from / (*rp->run_mats)->cols;
	int row_dest = to / (*rp->run_mats)->cols;

	if(row_src == row_dest)
		return 0;

	if(i8multimask(rp->run_mats, rp->mat_cnt, row_dest, to % (*rp->run_mats)->cols) & (1ULL << e))
		return 0;

	uint8_t sp, sn, dp, dn, se, de;
	near_streaks(rp, &sp, &sn, row_src, e, flag);
	near_streaks(rp, &dp, &dn, row_dest, e, flag);

	if(flag){
		se = MGET(rp->smat_up, row_src, e) & S_EMPTY;
		de = !(MGET(rp->smat_up, row_dest, e) & S_MANY);
	}else{
		se = !(MGET(rp->smat_up, row_src, e) & S_MANY);
		de = MGET(rp->smat_up, row_dest, e) & S_EMPTY;
	}

	if(row_src > row_dest){
		memswap_small(&sp, &sn, 8);
		memswap_small(&dp, &dn, 8);
	}

	uint8_t c = min(row_src, row_dest) + 1 + sn == max(row_src, row_dest);

	return (
		se ? ( de ? ( c ?
				(S(dn+1+sn, l) + S(sp, l) - S(dn, l) - S(dp, l)) :
				(S(sp, l) + S(sn, l) + S(dp+1+dn, l) - S(sp+1+sn, l) - S(dp, l) - S(dn, l))
			) : (S(sp, l) + S(sn, l) - S(sp+1+sn, l)) ) :
			( de ? 
				(S(dp+1+dn, l) - S(dp, l) - S(dn, l)) :
				(0))
	);
}

static int32_t streak_target(uint8_t x, uint8_t lambda){
	return x ? pow2(x - lambda) : 0;
}

static int32_t streak_decay(uint8_t x, uint8_t lambda){
	return  x ? (1 << lambda) >> (x-1) : 0;
}

static void near_streaks(struct runparms *rp, uint8_t *p, uint8_t *n, int row, uint8_t e, uint8_t flag){
	if(row > 0){
		uint8_t t = MGET(rp->smat_up, row-1, e);
		*p = ((t & S_EMPTY) == flag) * ((t & S_CMASK) + 1);
	}else{
		*p = 0;
	}

	if(row < (*rp->run_mats)->rows-1){
		uint8_t t = MGET(rp->smat_down, row+1, e);
		*n = ((t & S_EMPTY) == flag) * ((t & S_CMASK) + 1);
	}else{
		*n = 0;
	}
}

static int32_t cost_constv(struct runparms *rp, int from, int to, uint8_t e){
	if(e == EMPTY8)
		return 0;

	uint64_t ebit = 1ULL << e;
	int row_src = from / (*rp->run_mats)->cols;
	int row_dest = to / (*rp->run_mats)->cols;
	int32_t ret = 0;

	if(row_src != row_dest){
		if( (i8multimask(rp->run_mats, rp->mat_cnt, row_dest, to % (*rp->run_mats)->cols) & ebit) 
			? (MGET(rp->smat_up, row_dest, e) & S_MANY) :
			(!(MGET(rp->smat_up, row_dest, e) & S_EMPTY))) {

			ret++;
		}

		if(MGET(rp->smat_up, row_src, e) & S_MANY)
			ret--;
	}

	if(rp->cmat->data[from] & ebit)
		ret--;

	if(rp->cmat->data[to] & ebit)
		ret++;

	return ret;
}

static int rowcv(struct imat64 *cmat, struct imat8 **mats, int cnt, int row, uint64_t *mask){
	uint64_t m = 0;
	int ret = 0;
	for(int i=0;i<cnt;i++,mats++){
		uint8_t *rowp = MROW(*mats, row);
		for(int j=0;j<(*mats)->cols;j++){
			if(rowp[j] == EMPTY8)
				continue;
			assert(rowp[j] < 64);
			uint64_t bit = 1ULL << rowp[j];
			if(m & bit)
				ret++;
			if(cmat && (MGET(cmat, row, j) & bit))
				ret++;
			m |= bit;
		}
	}

	if(mask)
		*mask = m;

	return ret;
}

static void judge_init(struct runparms *rp, struct genparms *gp){
	rp_alloc_mats(rp, gp, gp->height, gp->arenas, 1);

	struct imat8 *mat = *rp->mats;

	uint8_t tmp[gp->elem_count];
	for(int i=0;i<gp->elem_count;i++)
		tmp[i] = i;

	rp->pop_cnt = 0;
	for(int i=0;i<mat->rows*mat->cols;i++){
		int idx = i % gp->elem_count;
		int e = tmp[idx];
		if(!(rp->cmat->data[i] & (1<<e))){
			mat->data[i] = e;
			rp->pop_cnt++;
		}else{
			for(int j=1;j<=gp->elem_count;j++){
				int s = j % gp->elem_count;
				if(!(rp->cmat->data[i] & (1 << tmp[s]))){
					mat->data[i] = tmp[s];
					memswap(&tmp[s], &tmp[idx], sizeof(uint8_t));
					rp->pop_cnt++;
					goto __continue__;
				}
			}
			mat->data[i] = EMPTY8;
		}

__continue__: ((void)0);
	}

}

static int judge_validate(struct runparms *rp, struct genparms *gp){
	return team_mat_validate(rp, gp);
}

static void team_init(struct runparms *rp, struct genparms *gp){
	assert(gp->bcount <= gp->elem_count);
	assert(gp->perfk == 1 || gp->perfk == 2);

	struct team_block blocks[gp->bcount];
	int perf_P, perf_S;

	team_block_init(blocks, gp, &perf_P, &perf_S);
	qsort(blocks, gp->bcount, sizeof(struct team_block), team_block_cmp);

	int t_max = team_calc_t_upper_bound(blocks, gp);
	rp_alloc_mats(rp, gp, t_max, gp->arenas, gp->perfk);
	assert((*rp->mats)->rows * (*rp->mats)->cols >= perf_P);

	for(int i=0;i<gp->bcount;i++){
		blocks[i].iter = 0;
		blocks[i].gen = 0;
	}

	int gen_P = 0;
	int t = 0;
	while(t < t_max && gen_P < perf_P){
		int c = 0;
		for(int i=0;i<gp->bcount;i++){
			int j_max = min(blocks[i].perf_s, blocks[i].perfcount - blocks[i].gen);
			for(int j=0;j<j_max;j++){
				if(gp->perfk == 2){
					team_gen_idx_k2(&MGET(rp->mats[0], t, c), &MGET(rp->mats[1], t, c), &blocks[i]);
				}else if(gp->perfk == 1){
					team_gen_idx_k1(&MGET(rp->mats[0], t, c), &blocks[i]);
				}else{
					assert(0);
				}
				if(++c == gp->arenas){
					c = 0;
					t++;
				}
				gen_P++;
			}
		}

		if(c){
			for(;c<gp->arenas;c++){
				for(int k=0;k<gp->perfk;k++)
					MSET(rp->mats[k], t, c, EMPTY8);
			}
			t++;
		}
	}

	assert(gp->ccnt || t == t_max);

	for(;t<t_max;t++){
		for(int i=0;i<gp->arenas;i++){
			for(int k=0;k<gp->perfk;k++)
				MSET(rp->mats[k], t, i, EMPTY8);
		}
	}

	assert(t == t_max);

	assert(gen_P == perf_P);
	rp->pop_cnt = gen_P;
}

static int team_calc_t_upper_bound(struct team_block *blocks, struct genparms *gp){
	int ret = 0, idx = 0, s = 0;
	for(int i=0;idx<gp->bcount;i++){
		if(!s)
			s = blocks[idx].perf_s;
		if(i%gp->arenas == 0)
			ret += blocks[idx].perfcount / blocks[idx].perf_s;
		if(!--s)
			idx++;
	}

	return ret + DIVCEIL(gp->ccnt, gp->arenas);
}

static void team_gen_idx_k2(uint8_t *idx0, uint8_t *idx1, struct team_block *b){
	int teams = b->end - b->start;
	int iter = 1 + b->iter/DIVCEIL(teams, 2);
	int j = b->iter%DIVCEIL(teams, 2);
	b->iter++;

	if(j > 0){
		int jj = CEIL2(teams) - 2;
		*idx0 = b->start + 1 + (iter + j - 1) % (jj + 1);
		*idx1 = b->start + 1 + (iter + jj - j) % (jj + 1);
	}else{
		*idx0 = b->start;
		*idx1 = b->start + iter;
	}

	if(*idx0 > *idx1)
		memswap(idx0, idx1, sizeof(int));

	assert(*idx1 > *idx0 && *idx0 >= b->start && *idx1 <= b->end);

	if(*idx1 == b->end)
		team_gen_idx_k2(idx0, idx1, b);
	else
		b->gen++;
}

static void team_gen_idx_k1(uint8_t *idx, struct team_block *b){
	*idx = b->start + (b->iter % (b->end - b->start));
	b->iter++;
	b->gen++;
}

static void team_block_init(struct team_block *p, struct genparms *gp, int *block_P, int *block_S){
	int *bsizes = gp->bsizes;
	if(!bsizes){
		bsizes = alloca(gp->bcount * sizeof(int));
		int t = DIVCEIL(gp->elem_count, gp->bcount);
		for(int i=0;i<gp->bcount;i++)
			bsizes[i] = t;
		bsizes[gp->bcount - 1] = gp->elem_count - t*(gp->bcount - 1);
	}

	int P = 0, S = 0;
	int team_cnt = 0;

	for(int i=0,eind=0;i<gp->bcount;i++){
		p[i].start = eind;
		eind += bsizes[i];
		p[i].end = eind;
		p[i].perf_s = bsizes[i] / gp->perfk;
		p[i].perfcount = gp->teamk ?
			(gp->teamk * p[i].perf_s) :
			(bsizes[i] * (int)ncr(bsizes[i] - gp->perfk + 1, gp->perfk - 1) / gp->perfk);

		P += p[i].perfcount;
		S += p[i].perf_s;
		team_cnt += bsizes[i];

		assert(bsizes[i] >= 0 && bsizes[i] <= gp->elem_count - gp->bcount + 1);
		assert(p[i].perfcount > 0);
		assert(p[i].perf_s > 0 && p[i].perf_s <= p[i].perfcount);
		assert((p[i].perfcount % p[i].perf_s) == 0);
	}

	assert(team_cnt == gp->elem_count);

	*block_P = P;
	*block_S = S;
}

static int team_mat_validate(struct runparms *rp, struct genparms *gp){
	struct imat8 **mats = rp->mats;
	int rows = (*mats)->rows;
	int cols = (*mats)->cols;

	for(int i=0;i<rows;i++){
		if(rowcv(NULL, mats, rp->mat_cnt, i, NULL)){
			fprintf(stderr, "Constraint violation: row %d: conflicting elements", i);
			return 1;
		}
		if(rowcv(rp->cmat, mats, rp->mat_cnt, i, NULL)){
			fprintf(stderr, "Constraint violation: row %d: forbidden element", i);
			return 4;
		}
	}

	if(rp->mat_cnt == 2){
		struct imat8 *temp = i8alloc(gp->elem_count, gp->elem_count);
		i8zero(temp);
		for(int i=0;i<rows*cols;i++){
			uint8_t a = min(mats[0]->data[i], mats[1]->data[i]);
			uint8_t b = max(mats[0]->data[i], mats[1]->data[i]);
			if(a == EMPTY8 || b == EMPTY8){
				if(a != EMPTY8 || b != EMPTY8){
					fprintf(stderr, "%d vs %d: no empty allowed", a, b);
					return 3;
				}
				continue;
			}
			int num = MGET(temp, a, b);
			if(num >= 1){
				fprintf(stderr, "%d vs %d is played too many times", a, b);
				return 2;
			}
			MSET(temp, a, b, num+1);
		}
		free(temp);
	}

	return 0;
}

static int team_block_cmp(const void *a, const void *b){
	return ((struct team_block *) b)->perfcount - ((struct team_block *) a)->perfcount;
}
